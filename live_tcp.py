# Live TCP receiver used by the QNTC demo.
#
# The sender transmits the same layout used by an offline 3DGStream scene:
# init_3dgs.ply first, then NTCs/config.json, then the NTC_*.pth files.
# The receiver writes those files into a cache and exposes them to the renderer
# as if they had already existed on disk. That keeps the live path close to the
# offline loader and avoids maintaining two separate scene formats.
#
# NTCs are loaded lazily. The viewer only loads the checkpoint needed for the
# frame being rendered, with a small cache and optional prefetching. Sparse
# update-rate modes are handled by holding the most recent available NTC for
# intermediate frames.

import os
import re
import json
import time
import socket
import struct
import shutil
import threading
import queue
from pathlib import Path
from collections import OrderedDict
import bisect

import util_gau

try:
    import torch
except Exception:
    torch = None

try:
    import tinycudann as tcnn
except Exception:
    tcnn = None

from NTC import NeuralTransformationCache

# Reuse the exact same compressed-checkpoint logic used by offline FVV loading.
# This keeps TCP live playback consistent with --autoload_fvv playback.
from util_3dgstream import (
    _extract_state_dict,
    _convert_fp16_tensors_to_fp32,
    _dequantize_int4_ntc_state_if_needed,
    _dequantize_int8_ntc_state_if_needed,
    _extract_bounds_from_ckpt,
)


def infer_total_frames(fvv_root: str) -> int:
    root = Path(fvv_root)
    ntc_dir = root / "NTCs"
    ntcs = sorted(ntc_dir.glob("NTC_*.pth"))
    return max(1, len(ntcs) + 1)


# Bigger chunks = fewer syscalls
_TCP_CHUNK = 4 * 1024 * 1024

_NTC_RE = re.compile(r"^NTCs[\\/]+NTC_(\d+)\.pth$", re.IGNORECASE)
# Accept both folder names (sender may use OFF fallback)
_ADD_RE = re.compile(r"^(additional_3dgs|additional_3dgs_OFF)[\\/]+additions_(\d+)\.ply$", re.IGNORECASE)


def _recvall(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError("socket closed")
        buf.extend(chunk)
    return bytes(buf)


def _safe_join(root: Path, rel: str) -> Path:
    """
    Secure join: prevents path traversal.
    Uses Path.relative_to() instead of startswith() (which can be bypassed).
    """
    rel = rel.replace("\\", "/").lstrip("/")
    rel = os.path.normpath(rel)
    if rel.startswith("..") or rel.startswith("../") or rel.startswith("..\\"):
        raise ValueError(f"invalid relative path: {rel}")

    root_res = root.resolve()
    p = (root_res / rel).resolve()
    try:
        p.relative_to(root_res)
    except Exception:
        raise ValueError(f"path escapes root: {rel}")
    return p


class _LazyNTCs:
    """
    List-like object: renderer.NTCs[t] returns a NeuralTransformationCache for frame t.
    Internally uses ONE persistent NTC instance and only swaps weights per index.
    """
    def __init__(self, live_state: "LiveTCPState"):
        self.live = live_state

    def __len__(self):
        # Return something plausible; renderer usually indexes directly.
        if self.live.total_frames_hint > 0:
            return self.live.total_frames_hint
        return max(1, (max(self.live.ntc_paths.keys(), default=-1) + 1))

    def __getitem__(self, idx: int):
        return self.live.get_ntc_for_index(int(idx))


class _LazyAdditions:
    """
    List-like object: renderer.additional_3dgs[t] returns additions for frame t.

    Important:
    For stable no-additions scenes, this object must behave like an empty list.
    Otherwise renderer_cuda.py may think additions exist and try to access .rot
    from None, causing:
      AttributeError("'NoneType' object has no attribute 'rot'")
    """
    def __init__(self, live_state: "LiveTCPState"):
        self.live = live_state

    def __len__(self):
        # No additions were received, so behave exactly like [].
        if not self.live.add_paths and not getattr(self.live, "_any_add_seen", False):
            return 0

        if self.live.total_frames_hint > 0:
            return self.live.total_frames_hint

        return max(0, (max(self.live.add_paths.keys(), default=-1) + 1))

    def __getitem__(self, idx: int):
        return self.live.get_add_for_index(int(idx))


class LiveTCPState:
    """
    FAST streaming state:
    - Receiver thread writes files to cache + enqueues rel_path
    - UI thread only updates maps and sets list-like providers for renderer
    - NTC is built ONCE and weights swapped per frame (lazy + cached)
    - Additions are lazy-loaded (and cached), not eagerly parsed
    """
    def __init__(
        self,
        cache_root: str,
        total_frames_hint: int = 0,
        autoplay: bool = False,
        verbose: bool = False,
        log_every_n_files: int = 30,
        ntc_state_cache: int = 4,
        add_cache: int = 8,
    ):
        self.cache_root = Path(cache_root)
        self.total_frames_hint = int(total_frames_hint) if total_frames_hint else 0
        self.autoplay = bool(autoplay)

        self.verbose = bool(verbose)
        self.log_every_n_files = max(1, int(log_every_n_files))

        # UI-visible status
        self.status = "idle"
        self.connected = ""
        self.err = ""

        self.files = 0
        self.total_bytes = 0

        # Application-level TCP stream monitor.
        # This is intentionally transport-agnostic: it measures what the receiver
        # actually writes to the cache, not kernel-level TCP internals.
        now_m = time.perf_counter()
        self._monitor_start_t = now_m
        self._last_rate_t = now_m
        self._last_rate_bytes = 0

        self.rx_mbps_avg = 0.0
        self.rx_mbps_inst = 0.0

        self.last_file = ""
        self.last_file_mb = 0.0
        self.last_file_ms = 0.0
        self.last_file_mbps = 0.0

        self.current_file = ""
        self.current_file_size = 0
        self.current_file_received = 0
        self.current_file_start_t = 0.0

        self.ntc_files = 0
        self.ntc_bytes = 0
        self._first_ntc_t = None
        self._last_ntc_t = None

        self.init_bytes = 0
        self.config_bytes = 0

        self.init_ply_ok = False
        self.config_ok = False

        # Instead of eagerly building lists, we map idx -> path (fast)
        self.ntc_paths = {}   # idx -> abs path
        self.add_paths = {}   # idx -> abs path
        self._any_add_seen = False

        # Renderer-facing list-like objects
        self.ntc_list = _LazyNTCs(self)
        self.add_list = _LazyAdditions(self)

        self.max_play = -1

        # ------------------------------------------------------------
        # Adaptive NTC update-rate / sparse NTC playback support
        # ------------------------------------------------------------
        # When the sender transmits only every N-th NTC, the receiver will
        # see sparse indices such as 0, 2, 4... or 0, 3, 6...
        # The renderer still asks for every frame index. In that case we
        # use "hold-last NTC": for frame t, use the newest received NTC j<=t.
        #
        # This keeps the viewer interactive in 3D: camera movement/parallax
        # still work, while only the temporal deformation is updated less often.
        self.hold_last_enabled = True

        # Sorted indices for fast nearest-left lookup.
        self._ntc_sorted_indices = []
        self._add_sorted_indices = []

        # Runtime diagnostics for the demo UI.
        self.active_ntc_idx = -1          # actual NTC used for the last rendered request
        self.last_requested_ntc_idx = -1  # requested frame/NTC index from renderer
        self.active_ntc_age = 0           # requested_idx - active_ntc_idx
        self.exact_ntc_uses = 0
        self.hold_last_ntc_uses = 0
        self.missing_ntc_uses = 0
        self.max_available_ntc = -1
        self.ntc_sparse_count = 0
        self.ntc_contiguous_count = 0
        self.inferred_ntc_stride = 1

        try:
            self._q = queue.SimpleQueue()
        except Exception:
            self._q = queue.Queue()

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

        # For NTC creation / bounds
        self.gaussians_loaded = False
        self._xyz_min = None
        self._xyz_max = None
        self._ntc_conf = None
        self._base_idx = None

        # Detect renderer mode
        self._cuda_mode = None

        # One persistent NTC + weight swapping
        self._ntc_single = None
        self._ntc_loaded_idx = None
        self._ntc_lock = threading.Lock()
        self._ntc_state_cache_cap = max(0, int(ntc_state_cache))
        self._ntc_state_cache = OrderedDict()  # idx -> state dict (CPU)

        # Lazy additions cache
        self._add_cache_cap = max(0, int(add_cache))
        self._add_cache = OrderedDict()  # idx -> add_obj (CPU or CUDA)

        # Lightweight prefetch (optional, safe default ON)
        self._prefetch_q = queue.Queue(maxsize=32)
        self._prefetch_inflight = set()
        self._prefetch_thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._prefetch_thread.start()

        # receiver stats
        self._last_log_files = 0

    def request_stop(self):
        self._stop.set()

    # ---------- Receiver (fast I/O) ----------

    def start_receiver(self, bind_host: str, port: int, clear_cache: bool):
        if clear_cache and self.cache_root.exists():
            shutil.rmtree(self.cache_root, ignore_errors=True)

        (self.cache_root / "NTCs").mkdir(parents=True, exist_ok=True)
        (self.cache_root / "additional_3dgs").mkdir(parents=True, exist_ok=True)
        (self.cache_root / "additional_3dgs_OFF").mkdir(parents=True, exist_ok=True)

        def _run():
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
            except Exception:
                pass

            srv.bind((bind_host, port))
            srv.listen(1)

            with self._lock:
                self.status = f"listening {bind_host}:{port}"
                self.err = ""

            if self.verbose:
                print(f"[RECV] listening on {bind_host}:{port}")

            srv.settimeout(0.5)
            conn = None
            addr = None
            while not self._stop.is_set():
                try:
                    conn, addr = srv.accept()
                    break
                except socket.timeout:
                    continue

            if self._stop.is_set():
                try:
                    srv.close()
                except Exception:
                    pass
                return

            with self._lock:
                self.status = "connected"
                self.connected = f"{addr[0]}:{addr[1]}"
                self.err = ""

            if self.verbose:
                print(f"[RECV] connected from {addr[0]}:{addr[1]}")

            try:
                with conn:
                    try:
                        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    except Exception:
                        pass
                    try:
                        conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
                    except Exception:
                        pass
                    conn.settimeout(1.0)

                    buf = bytearray(_TCP_CHUNK)

                    while not self._stop.is_set():
                        try:
                            hdr = _recvall(conn, 4)
                        except socket.timeout:
                            continue
                        except EOFError:
                            break

                        (path_len,) = struct.unpack("!I", hdr)
                        if path_len <= 0 or path_len > 1024 * 16:
                            raise RuntimeError(f"bad path_len={path_len}")

                        rel_path = _recvall(conn, path_len).decode("utf-8", errors="replace")
                        (size,) = struct.unpack("!Q", _recvall(conn, 8))

                        if rel_path in ("END", "__END__"):
                            break

                        out_path = _safe_join(self.cache_root, rel_path)
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        tmp_path = out_path.with_suffix(out_path.suffix + ".part")

                        remaining = size
                        written = 0

                        t0 = time.perf_counter()
                        with self._lock:
                            self.current_file = rel_path
                            self.current_file_size = int(size)
                            self.current_file_received = 0
                            self.current_file_start_t = t0
                        try:
                            with open(tmp_path, "wb", buffering=1024 * 1024) as f:
                                mv = memoryview(buf)
                                while remaining > 0:
                                    n = _TCP_CHUNK if remaining >= _TCP_CHUNK else int(remaining)
                                    view = mv[:n]
                                    try:
                                        got = conn.recv_into(view)
                                    except socket.timeout:
                                        continue
                                    if got <= 0:
                                        raise EOFError("socket closed mid-file")
                                    f.write(view[:got])
                                    remaining -= got
                                    written += got
                                    with self._lock:
                                        self.current_file_received = int(written)
                            os.replace(tmp_path, out_path)
                        finally:
                            if tmp_path.exists() and not out_path.exists():
                                try:
                                    tmp_path.unlink()
                                except Exception:
                                    pass

                        dt_ms = (time.perf_counter() - t0) * 1000.0
                        file_mbps = (written * 8.0 / 1_000_000.0) / max(1e-6, dt_ms / 1000.0)
                        now_rate = time.perf_counter()

                        rel_norm = rel_path.replace("\\", "/").lower()
                        is_ntc_file = _NTC_RE.match(rel_path.replace("/", "\\")) is not None
                        is_init_file = rel_norm == "init_3dgs.ply"
                        is_config_file = rel_norm.endswith("ntcs/config.json") or rel_norm == "config.json"

                        with self._lock:
                            self.files += 1
                            self.total_bytes += written

                            elapsed = max(1e-6, now_rate - self._monitor_start_t)
                            self.rx_mbps_avg = (self.total_bytes * 8.0 / 1_000_000.0) / elapsed

                            delta_t = max(1e-6, now_rate - self._last_rate_t)
                            delta_b = self.total_bytes - self._last_rate_bytes
                            self.rx_mbps_inst = (delta_b * 8.0 / 1_000_000.0) / delta_t
                            self._last_rate_t = now_rate
                            self._last_rate_bytes = self.total_bytes

                            self.last_file = rel_path
                            self.last_file_mb = written / (1024.0 * 1024.0)
                            self.last_file_ms = dt_ms
                            self.last_file_mbps = file_mbps

                            self.current_file = ""
                            self.current_file_size = 0
                            self.current_file_received = 0
                            self.current_file_start_t = 0.0

                            if is_ntc_file:
                                self.ntc_files += 1
                                self.ntc_bytes += written
                                if self._first_ntc_t is None:
                                    self._first_ntc_t = now_rate
                                self._last_ntc_t = now_rate

                            if is_init_file:
                                self.init_bytes += written

                            if is_config_file:
                                self.config_bytes += written

                        # logging throttled (printing is slow)
                        if self.verbose and (self.files % self.log_every_n_files == 0):
                            mb = written / (1024.0 * 1024.0)
                            print(f"[RECV] #{self.files} last={rel_path} ({mb:.2f} MB) {dt_ms:.1f} ms")

                        self._q.put(rel_path)

            except Exception as e:
                with self._lock:
                    self.status = "error"
                    self.err = repr(e)
                if self.verbose:
                    print("[RECV] ERROR:", repr(e))

            with self._lock:
                if self.status != "error":
                    self.status = "done"

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    # ---------- NTC build / swap (FAST) ----------

    def _torch_device(self) -> str:
        if torch is None:
            raise RuntimeError("torch not available")
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_ntc_conf(self):
        if self._ntc_conf is not None:
            return
        cfg_path = self.cache_root / "NTCs" / "config.json"
        if not cfg_path.exists():
            raise RuntimeError("NTCs/config.json not received yet")
        with open(cfg_path, "r", encoding="utf-8") as f:
            self._ntc_conf = json.load(f)
        if ("encoding" not in self._ntc_conf) or ("network" not in self._ntc_conf):
            raise RuntimeError("Invalid NTC config.json (missing 'encoding'/'network')")
        self.config_ok = True

    def _ensure_bounds(self, renderer):
        if self._xyz_min is not None and self._xyz_max is not None:
            return

        # Prefer the first received NTC checkpoint bounds, matching util_3dgstream.load_NTCs().
        # Recomputing bounds from the Gaussian cloud can make NTC deformation incorrect.
        first_path = None
        if 0 in self.ntc_paths:
            first_path = self.ntc_paths[0]
        elif self.ntc_paths:
            first_key = sorted(self.ntc_paths.keys())[0]
            first_path = self.ntc_paths[first_key]

        if first_path is not None:
            try:
                ckpt = self._load_raw_checkpoint(first_path)
                dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                xyz_min, xyz_max = _extract_bounds_from_ckpt(ckpt, dev)
                if xyz_min is not None and xyz_max is not None:
                    self._xyz_min = xyz_min
                    self._xyz_max = xyz_max
                    print("[LIVE NTC BOUNDS] using checkpoint xyz_bound_min / xyz_bound_max", flush=True)
                    print("[LIVE NTC BOUNDS] xyz_min =", xyz_min.detach().cpu().numpy(), flush=True)
                    print("[LIVE NTC BOUNDS] xyz_max =", xyz_max.detach().cpu().numpy(), flush=True)
                    return
            except Exception as e:
                with self._lock:
                    self.err = f"live checkpoint bounds fallback: {repr(e)}"

        # Fallback only if checkpoint bounds are not available.
        if not hasattr(renderer, "gaussians") or renderer.gaussians is None:
            raise RuntimeError("renderer.gaussians not ready for bounds")
        if not hasattr(renderer.gaussians, "get_xyz_bound"):
            raise RuntimeError("renderer.gaussians missing get_xyz_bound()")

        xyz_min, xyz_max = renderer.gaussians.get_xyz_bound()
        self._xyz_min = xyz_min
        self._xyz_max = xyz_max
        print("[LIVE NTC BOUNDS] fallback to Gaussian quantile bounds", flush=True)
    def _ensure_cuda_mode(self, renderer):
        if self._cuda_mode is not None:
            return
        cuda_mode = False
        if torch is not None and hasattr(renderer, "gaussians") and hasattr(renderer.gaussians, "xyz"):
            if isinstance(renderer.gaussians.xyz, torch.Tensor):
                cuda_mode = bool(renderer.gaussians.xyz.is_cuda)
        self._cuda_mode = cuda_mode

    def _ensure_ntc_single(self, renderer):
        if self._ntc_single is not None:
            return
        if torch is None:
            raise RuntimeError("torch not available")
        if tcnn is None:
            raise RuntimeError("tinycudann not available")

        dev = self._torch_device()
        if dev != "cuda":
            raise RuntimeError("NTC requires CUDA (tinycudann). torch.cuda.is_available() is False")

        self._ensure_ntc_conf()
        self._ensure_bounds(renderer)

        model = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=8,
            encoding_config=self._ntc_conf["encoding"],
            network_config=self._ntc_conf["network"],
        ).cuda()

        ntc = NeuralTransformationCache(model, self._xyz_min, self._xyz_max).cuda()
        ntc.eval()

        self._ntc_single = ntc
        self._ntc_loaded_idx = None

    def _cache_put(self, od: OrderedDict, key, val, cap: int):
        od[key] = val
        od.move_to_end(key)
        while cap > 0 and len(od) > cap:
            od.popitem(last=False)

    def _load_raw_checkpoint(self, path: str):
        # Try weights_only if available (torch>=2), else normal.
        if torch is None:
            raise RuntimeError("torch not available")
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def _load_state_dict_fast(self, path: str):
        """
        Load an NTC checkpoint and convert compressed formats into model.params.

        This mirrors util_3dgstream.load_NTCs():
          raw checkpoint
          extract state dict
          INT4 dequantize if needed
          INT8 dequantize if needed
          FP16 -> FP32 if needed

        Without this, live TCP mode passes raw qint4/qint8 checkpoints directly
        into tiny-cuda-nn with strict=False, which can silently skip model.params.
        """
        ckpt = self._load_raw_checkpoint(path)

        state = _extract_state_dict(ckpt)
        state = _dequantize_int4_ntc_state_if_needed(state)
        state = _dequantize_int8_ntc_state_if_needed(state)
        state = _convert_fp16_tensors_to_fp32(state)

        if not isinstance(state, dict) or "model.params" not in state:
            raise RuntimeError(
                f"NTC checkpoint did not produce model.params after decompression: {path}"
            )

        return state

    def _prefetch_loop(self):
        while True:
            idx = self._prefetch_q.get()
            if idx is None:
                return
            try:
                # NTC prefetch
                p = self.ntc_paths.get(idx)
                if p and self._ntc_state_cache_cap > 0:
                    if idx not in self._ntc_state_cache:
                        sd = self._load_state_dict_fast(p)
                        self._cache_put(self._ntc_state_cache, idx, sd, self._ntc_state_cache_cap)
                # Additions prefetch (CPU only; CUDA conversion happens on demand in UI thread)
                ap = self.add_paths.get(idx)
                if ap and self._add_cache_cap > 0 and idx not in self._add_cache:
                    add_cpu = util_gau.load_ply(ap)
                    self._cache_put(self._add_cache, idx, add_cpu, self._add_cache_cap)
            except Exception:
                pass
            finally:
                try:
                    self._prefetch_inflight.discard(idx)
                except Exception:
                    pass

    def _request_prefetch(self, idx: int):
        if idx < 0:
            return
        if idx in self._prefetch_inflight:
            return
        if self._prefetch_q.full():
            return
        self._prefetch_inflight.add(idx)
        try:
            self._prefetch_q.put_nowait(idx)
        except Exception:
            self._prefetch_inflight.discard(idx)


    # ---------- Sparse / adaptive update-rate helpers ----------

    def _insert_sorted_unique(self, arr, idx: int):
        """Insert idx into sorted list arr if it is not already present."""
        idx = int(idx)
        pos = bisect.bisect_left(arr, idx)
        if pos >= len(arr) or arr[pos] != idx:
            arr.insert(pos, idx)

    def _infer_ntc_stride_locked(self) -> int:
        """
        Estimate the sender stride from received NTC indices.

        Examples:
          [0,1,2,3]     -> stride 1
          [0,2,4,6]     -> stride 2
          [0,3,6,9]     -> stride 3

        This is only used to estimate how far playback can safely advance
        after the newest received NTC when hold-last is enabled.
        """
        arr = self._ntc_sorted_indices
        if len(arr) < 2:
            return 1

        diffs = []
        prev = arr[0]
        for cur in arr[1:]:
            d = int(cur) - int(prev)
            if d > 0:
                diffs.append(d)
            prev = cur

        if not diffs:
            return 1

        # Use the most common recent positive spacing.
        # This is robust enough for stride-2/3/5 demo modes.
        recent = diffs[-16:]
        counts = {}
        for d in recent:
            counts[d] = counts.get(d, 0) + 1
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def _resolve_available_ntc_index(self, requested_idx: int):
        """
        Resolve renderer request -> actual NTC index.

        Returns:
          (actual_idx, exact)
            actual_idx: requested index if available, otherwise latest received <= requested.
            exact: True if actual_idx == requested_idx.

        If nothing usable exists yet, returns (None, False).
        """
        requested_idx = int(requested_idx)

        if requested_idx in self.ntc_paths:
            return requested_idx, True

        if not self.hold_last_enabled:
            return None, False

        arr = self._ntc_sorted_indices
        if not arr:
            return None, False

        pos = bisect.bisect_right(arr, requested_idx) - 1
        if pos < 0:
            return None, False

        return int(arr[pos]), False

    def _next_known_ntc_after(self, idx: int):
        """Return next received NTC index after idx, or None."""
        arr = self._ntc_sorted_indices
        pos = bisect.bisect_right(arr, int(idx))
        if pos < len(arr):
            return int(arr[pos])
        return None

    def get_ntc_for_index(self, idx: int):
        """
        Called by renderer during draw: returns the NTC object for a given timestep.

        Adaptive/stride behavior:
        - If the exact NTC idx exists, use it.
        - If it does not exist, use the latest received NTC <= idx.
          This is the "hold-last NTC" mode.
        - This allows sender-side temporal subsampling such as:
              0,2,4,6...   or   0,3,6,9...
          while the viewer still renders every display frame.
        """
        if not (self.init_ply_ok and self.config_ok):
            return None

        requested_idx = int(idx)
        actual_idx, exact = self._resolve_available_ntc_index(requested_idx)

        if actual_idx is None:
            with self._lock:
                self.last_requested_ntc_idx = requested_idx
                self.active_ntc_idx = -1
                self.active_ntc_age = 0
                self.missing_ntc_uses += 1
            return None

        path = self.ntc_paths.get(actual_idx)
        if path is None:
            with self._lock:
                self.last_requested_ntc_idx = requested_idx
                self.active_ntc_idx = -1
                self.active_ntc_age = 0
                self.missing_ntc_uses += 1
            return None

        with self._lock:
            self.last_requested_ntc_idx = requested_idx
            self.active_ntc_idx = int(actual_idx)
            self.active_ntc_age = max(0, requested_idx - int(actual_idx))
            if exact:
                self.exact_ntc_uses += 1
            else:
                self.hold_last_ntc_uses += 1

        with self._ntc_lock:
            try:
                self._ensure_ntc_single(self._renderer_ref)
            except Exception as e:
                with self._lock:
                    self.err = f"NTC init error: {repr(e)}"
                return None

            if self._ntc_loaded_idx != actual_idx:
                # state dict from cache or disk
                state = self._ntc_state_cache.get(actual_idx)
                if state is None:
                    try:
                        state = self._load_state_dict_fast(path)
                    except Exception as e:
                        with self._lock:
                            self.err = f"load NTC_{actual_idx:06d}.pth: {repr(e)}"
                        return None
                    if self._ntc_state_cache_cap > 0:
                        self._cache_put(self._ntc_state_cache, actual_idx, state, self._ntc_state_cache_cap)

                try:
                    self._ntc_single.load_state_dict(state, strict=False)
                    self._ntc_loaded_idx = actual_idx
                except Exception as e:
                    with self._lock:
                        self.err = f"apply NTC_{actual_idx:06d}.pth: {repr(e)}"
                    return None

                # Prefetch the next received NTC if known. If not known yet,
                # the file-ingest path will prefetch when it arrives.
                next_idx = self._next_known_ntc_after(actual_idx)
                if next_idx is not None:
                    self._request_prefetch(next_idx)
                else:
                    self._request_prefetch(actual_idx + 1)

            return self._ntc_single

    def get_add_for_index(self, idx: int):
        """
        Called by renderer: returns additions for timestep idx, or None.
        Lazy loads + caches. In CUDA mode, never returns CPU-only data.
        """
        if not self.init_ply_ok:
            return None

        path = self.add_paths.get(idx)
        if path is None:
            return None

        # cached?
        add_obj = self._add_cache.get(idx)
        if add_obj is not None:
            # Prefetch next (helps sequential)
            self._request_prefetch(idx + 1)
            # If CUDA mode and cache contains CPU object, convert on demand
            if self._cuda_mode:
                try:
                    from renderer_cuda import gaus_cuda_from_cpu
                    if not (torch is not None and isinstance(getattr(self._renderer_ref.gaussians, "xyz", None), torch.Tensor)):
                        return None
                    # Convert CPU->CUDA (one-time)
                    add_cuda = gaus_cuda_from_cpu(add_obj)
                    self._cache_put(self._add_cache, idx, add_cuda, self._add_cache_cap)
                    return add_cuda
                except Exception as e:
                    with self._lock:
                        self.err = f"CUDA convert additions_{idx:06d}.ply: {repr(e)}"
                    return None
            return add_obj

        # Not cached: load now (can be heavy)
        try:
            add_cpu = util_gau.load_ply(path)
        except Exception as e:
            with self._lock:
                self.err = f"load additions_{idx:06d}.ply: {repr(e)}"
            return None

        if self._add_cache_cap > 0:
            self._cache_put(self._add_cache, idx, add_cpu, self._add_cache_cap)

        # Prefetch next
        self._request_prefetch(idx + 1)

        if self._cuda_mode:
            try:
                from renderer_cuda import gaus_cuda_from_cpu
                add_cuda = gaus_cuda_from_cpu(add_cpu)
                if self._add_cache_cap > 0:
                    self._cache_put(self._add_cache, idx, add_cuda, self._add_cache_cap)
                return add_cuda
            except Exception as e:
                with self._lock:
                    self.err = f"CUDA convert additions_{idx:06d}.ply: {repr(e)}"
                return None

        return add_cpu

    # ---------- UI thread ingest (FAST) ----------

    def process_new_files(self, renderer, camera):
        """
        Called every UI tick.
        Goal: be extremely cheap.
        """
        # Keep a reference so lazy getters can access renderer
        self._renderer_ref = renderer

        # process queue with a small time budget to avoid UI stutter
        t_budget_ms = 3.0
        t0 = time.perf_counter()

        while True:
            # time budget
            if (time.perf_counter() - t0) * 1000.0 > t_budget_ms:
                break

            try:
                rel_path = self._q.get_nowait()
            except Exception:
                break

            abs_path = (self.cache_root / rel_path).resolve()
            rel_norm = rel_path.replace("\\", "/")

            if rel_norm == "init_3dgs.ply":
                try:
                    gaussians = util_gau.load_ply(str(abs_path))
                    renderer.update_gaussian_data(gaussians)
                    renderer.sort_and_update(camera)

                    self.gaussians_loaded = True
                    self.init_ply_ok = True

                    # bounds must be recomputed from real gaussians
                    self._xyz_min = None
                    self._xyz_max = None

                    # detect renderer mode
                    self._ensure_cuda_mode(renderer)

                    # attach lazy providers
                    renderer.NTCs = self.ntc_list
                    renderer.additional_3dgs = self.add_list

                    # if config already there, create the single NTC now
                    if self.config_ok:
                        try:
                            self._ensure_ntc_single(renderer)
                        except Exception as e:
                            with self._lock:
                                self.err = f"NTC init error: {repr(e)}"

                except Exception as e:
                    with self._lock:
                        self.err = f"load init_3dgs.ply: {repr(e)}"
                continue

            if rel_norm.lower() == "ntcs/config.json":
                try:
                    self._ntc_conf = None
                    self._ensure_ntc_conf()
                    # if init already there, create NTC once now
                    if self.init_ply_ok:
                        try:
                            self._ensure_ntc_single(renderer)
                        except Exception as e:
                            with self._lock:
                                self.err = f"NTC init error: {repr(e)}"
                except Exception as e:
                    with self._lock:
                        self.err = f"load NTCs/config.json: {repr(e)}"
                continue

            m_ntc = _NTC_RE.match(rel_path.replace("/", "\\"))
            if m_ntc:
                idx_real = int(m_ntc.group(1))
                if self._base_idx is None:
                    self._base_idx = idx_real
                idx = idx_real - self._base_idx
                if idx < 0:
                    idx = idx_real
                self.ntc_paths[idx] = str(abs_path)
                self._insert_sorted_unique(self._ntc_sorted_indices, idx)
                # Prefetch soon-ish
                self._request_prefetch(idx)
                continue

            m_add = _ADD_RE.match(rel_path.replace("/", "\\"))
            if m_add:
                idx_real = int(m_add.group(2))
                if self._base_idx is None:
                    self._base_idx = idx_real
                idx = idx_real - self._base_idx
                if idx < 0:
                    idx = idx_real
                self.add_paths[idx] = str(abs_path)
                self._insert_sorted_unique(self._add_sorted_indices, idx)
                self._any_add_seen = True
                self._request_prefetch(idx)
                continue

        # Update playback availability.
        #
        # Old behavior required contiguous NTC indices from 0, which breaks
        # stride modes such as 0,2,4... or 0,3,6...
        #
        # New behavior:
        #   max_available_ntc = newest received NTC index.
        #   inferred_stride = spacing between received NTCs.
        #   max_play = newest index + stride - 1, capped by total_frames_hint.
        #
        # Example stride=3, received NTCs [0,3,6]:
        #   max_available_ntc = 6
        #   max_play = 8
        #   frames 7 and 8 use hold-last NTC 6.
        k_contig = 0
        while k_contig in self.ntc_paths:
            k_contig += 1

        ntc_total = len(self.ntc_paths)
        max_ntc = max(self.ntc_paths.keys(), default=-1)
        inferred_stride = self._infer_ntc_stride_locked() if ntc_total > 0 else 1
        self.ntc_sparse_count = ntc_total
        self.ntc_contiguous_count = k_contig
        self.max_available_ntc = max_ntc
        self.inferred_ntc_stride = max(1, int(inferred_stride))

        if max_ntc >= 0:
            sparse_max_play = max_ntc + max(0, self.inferred_ntc_stride - 1)
            if self.total_frames_hint > 0:
                sparse_max_play = min(sparse_max_play, self.total_frames_hint - 1)
        else:
            sparse_max_play = -1

        if self._any_add_seen:
            # If additions are used, be conservative. Most of your stable QNTC scenes
            # have no additions, so this branch is normally inactive.
            max_add = max(self.add_paths.keys(), default=-1)
            if max_add >= 0:
                self.max_play = min(sparse_max_play, max_add)
            else:
                self.max_play = -1
        else:
            self.max_play = sparse_max_play

        with self._lock:
            if (
                self.status == "connected"
                and self.max_play >= 0
                and self.init_ply_ok
                and self.config_ok
            ):
                self.status = "streaming"

    def ui_snapshot(self):
        # Sparse playback-aware counters.
        # ready_ntc now means total NTCs received, not contiguous-only.
        k_contig = 0
        while k_contig in self.ntc_paths:
            k_contig += 1

        ntc_total = len(self.ntc_paths)
        max_ntc = max(self.ntc_paths.keys(), default=-1)

        a_contig = 0
        while a_contig in self.add_paths:
            a_contig += 1
        add_total = len(self.add_paths)

        now_m = time.perf_counter()

        with self._lock:
            gb = self.total_bytes / (1024.0 * 1024.0 * 1024.0)
            total_mb = self.total_bytes / (1024.0 * 1024.0)

            current_progress = 0.0
            current_file_mbps = 0.0
            if self.current_file_size > 0:
                current_progress = self.current_file_received / max(1, self.current_file_size)
                cur_elapsed = max(1e-6, now_m - self.current_file_start_t)
                current_file_mbps = (
                    self.current_file_received * 8.0 / 1_000_000.0
                ) / cur_elapsed

            ntc_mb = self.ntc_bytes / (1024.0 * 1024.0)
            ntc_transfer_done = (
                (self.status == "done" and self.ntc_files > 0)
                or (
                    self.total_frames_hint > 0
                    and self.ntc_files >= max(0, self.total_frames_hint - 1)
                )
            )

            if self._first_ntc_t is not None:
                if ntc_transfer_done and self._last_ntc_t is not None:
                    # Freeze the measured NTC delivery rate after all NTCs arrive.
                    # Otherwise the displayed average decays while the user keeps watching.
                    ntc_elapsed = max(1e-6, self._last_ntc_t - self._first_ntc_t)
                else:
                    ntc_elapsed = max(1e-6, now_m - self._first_ntc_t)

                ntc_mbps_avg = (self.ntc_bytes * 8.0 / 1_000_000.0) / ntc_elapsed
                ntc_fps_recv = self.ntc_files / ntc_elapsed
            else:
                ntc_mbps_avg = 0.0
                ntc_fps_recv = 0.0

            return {
                "cache": str(self.cache_root),
                "status": self.status,
                "connected": self.connected,
                "err": self.err,
                "init_ply": self.init_ply_ok,
                "config": self.config_ok,
                "ready_ntc": ntc_total,
                "ready_ntc_contiguous": k_contig,
                "ready_ntc_total": ntc_total,
                "ready_add": add_total,
                "ready_add_contiguous": a_contig,
                "max_play": self.max_play,
                "max_available_ntc": self.max_available_ntc,
                "inferred_ntc_stride": self.inferred_ntc_stride,

                # Hold-last diagnostics.
                "hold_last_enabled": self.hold_last_enabled,
                "last_requested_ntc_idx": self.last_requested_ntc_idx,
                "active_ntc_idx": self.active_ntc_idx,
                "active_ntc_age": self.active_ntc_age,
                "exact_ntc_uses": self.exact_ntc_uses,
                "hold_last_ntc_uses": self.hold_last_ntc_uses,
                "missing_ntc_uses": self.missing_ntc_uses,

                "files": self.files,
                "gb": gb,
                "total_mb": total_mb,
                "base_idx": self._base_idx,

                # Stream monitor fields.
                "rx_mbps_avg": self.rx_mbps_avg,
                "rx_mbps_inst": self.rx_mbps_inst,
                "last_file": self.last_file,
                "last_file_mb": self.last_file_mb,
                "last_file_ms": self.last_file_ms,
                "last_file_mbps": self.last_file_mbps,
                "current_file": self.current_file,
                "current_file_size": self.current_file_size,
                "current_file_received": self.current_file_received,
                "current_progress": current_progress,
                "current_file_mbps": current_file_mbps,
                "ntc_files": self.ntc_files,
                "ntc_mb": ntc_mb,
                "ntc_mbps_avg": ntc_mbps_avg,
                "ntc_fps_recv": ntc_fps_recv,
                "ntc_transfer_done": ntc_transfer_done,
                "init_mb": self.init_bytes / (1024.0 * 1024.0),
                "config_kb": self.config_bytes / 1024.0,
            }
