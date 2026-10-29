"""Progressive INT4 NTC receiver, bounded CPU cache, real QUIC stage files.

Drop-in replacement for the experimental qntc_progressive_live.py sidecar.
Original Gaussians / colors and original compressed NTCs remain untouched.
A CPU-only worker decodes only near-playhead NTCs; the UI draws with the
most recently completed local stage snapshot. No CUDA from worker thread.
"""
from __future__ import annotations

import json
import os
import queue
import re
import struct
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from live_quic import LiveQUICState as FileLiveQUICState
from qntc_progressive_wire import BLOCK, HASH, TOTAL, HEADER, MAGIC

_STAGENAME = re.compile(r"^NTC_(\d{6})_P([0-3])\.qblk$")
_RECORD_DTYPE = np.dtype([
    ("idx", ">u4"), ("scale", ">f4"), ("packed", "u1", (32,))
])
assert _RECORD_DTYPE.itemsize == 40


def decode_stages(paths):
    """Rebuild one partially received NTC on CPU with vectorized nibble decode.

    paths: sequence of (stage_number, path) pairs. Each file has already
    been atomically committed by the existing QUIC file transport.
    """
    params = np.zeros((TOTAL, BLOCK), dtype=np.float32)
    seen = np.zeros(TOTAL, dtype=np.bool_)
    frame_seen = None
    nblocks = 0
    for stage_expected, path in sorted(paths):
        raw_bytes = Path(path).read_bytes()
        if len(raw_bytes) < HEADER.size:
            raise ValueError(f"short stage file: {path}")
        magic, frame, stage, expected, count = HEADER.unpack_from(raw_bytes)
        if (magic != MAGIC or stage != stage_expected or expected != 1556
                or count > 1556 or len(raw_bytes) != HEADER.size + 40*count):
            raise ValueError(f"invalid stage header/size: {path}")
        if frame_seen is None:
            frame_seen = frame
        elif frame_seen != frame:
            raise ValueError("mixed NTC frames")
        rec = np.frombuffer(raw_bytes, dtype=_RECORD_DTYPE, count=count, offset=HEADER.size)
        ids = rec["idx"].astype(np.int64)
        scales = rec["scale"].astype(np.float32)
        if (np.any(ids >= TOTAL) or np.any(seen[ids]) or
            len(np.unique(ids)) != len(ids) or
            np.any(~np.isfinite(scales)) or np.any(scales < 0)):
            raise ValueError("bad scale, repeated ID or out-of-range block")
        seen[ids] = True
        packed = rec["packed"]
        values = np.empty((count, BLOCK), dtype=np.float32)
        values[:, 0::2] = (packed & 15).astype(np.float32) - 8.0
        values[:, 1::2] = (packed >> 4).astype(np.float32) - 8.0
        params[ids] = values * scales[:, None]
        nblocks += count
    if len(paths) == 4 and nblocks != 1556:
        raise ValueError(f"full NTC contains {nblocks} records instead of 1556")
    return frame_seen, params, nblocks


class LiveQUICState(FileLiveQUICState):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hold_last_enabled = False
        self.part_versions = {}        # IMPORTANT: published only after CPU decode
        self._stage_files = {}         # frame -> stage -> committed path
        self._seen_stage_names = set()
        self._prepared = OrderedDict() # frame -> (version, 8 MiB float32 CPU state)
        self._ever_ready = set()
        self._queued = set()
        self._work = queue.Queue(maxsize=32)
        self._prep_lock = threading.Lock()
        self._cache_cap = max(4, min(16, int(os.environ.get("QNTC_PROGRESSIVE_CPU_CACHE", "10"))))
        self._ahead = max(1, min(6, int(os.environ.get("QNTC_PROGRESSIVE_AHEAD", "3"))))
        self._progressive_path = self.cache_root / 'NTCs/progressive'
        self._bounds_path = self._progressive_path / 'bounds.json'
        self._scan_at = 0.0
        self._progressive_loaded = None
        self._decode_thread = threading.Thread(target=self._decode_loop, daemon=True,
                                                name="qntc-progressive-cpu")
        self._decode_thread.start()

    def _ensure_bounds(self, renderer):
        if self._xyz_min is not None and self._xyz_max is not None:
            return
        if not self._bounds_path.is_file():
            raise RuntimeError('progressive bounds.json must arrive before initialization')
        data = json.loads(self._bounds_path.read_text())
        lo = torch.tensor(data['xyz_bound_min'], device='cuda', dtype=torch.float32)
        hi = torch.tensor(data['xyz_bound_max'], device='cuda', dtype=torch.float32)
        if (tuple(lo.shape) != (3,) or tuple(hi.shape) != (3,)
                or not bool(torch.all(hi > lo).item())):
            raise ValueError('invalid progressive NTC bounds')
        self._xyz_min, self._xyz_max = lo, hi
        print('[REAL QUIC] NTC bounds loaded', flush=True)

    def _enqueue(self, frame):
        with self._prep_lock:
            files = self._stage_files.get(frame)
            if not files or frame in self._queued:
                return
            current = self._prepared.get(frame)
            if current is not None and current[0] == len(files):
                return
            self._queued.add(frame)
        try:
            self._work.put_nowait(frame)
        except queue.Full:
            with self._prep_lock:
                self._queued.discard(frame)

    def _decode_loop(self):
        while True:
            frame = self._work.get()
            with self._prep_lock:
                self._queued.discard(frame)
                files = sorted(self._stage_files.get(frame, {}).items())
                current_version = len(files)
                current = self._prepared.get(frame)
                if current is not None and current[0] == current_version:
                    continue
            try:
                decoded_frame, params, count = decode_stages(files)
                if decoded_frame != frame:
                    raise ValueError(f'header frame {decoded_frame} != filename frame {frame}')
            except Exception as exc:
                self.err = f'QUIC decode frame {frame}: {exc!r}'
                print('[REAL QUIC ERROR]', self.err, flush=True)
                continue
            requeue = False
            with self._prep_lock:
                if len(self._stage_files.get(frame, {})) != current_version:
                    requeue = True
                else:
                    self._prepared[frame] = (current_version, params)
                    self._prepared.move_to_end(frame)
                    self.part_versions[frame] = current_version
                    self._ever_ready.add(frame)
                    self._trim_cache_locked()
            if requeue:
                self._enqueue(frame)
            else:
                print(f'[QUIC CPU READY] NTC={frame:03d} stages={current_version}/4 '
                      f'blocks={count}/1556 cache={len(self._prepared)}', flush=True)

    def _trim_cache_locked(self):
        # Prefer dropping snapshots outside the playhead window. Compact QUIC
        # stage files remain on disk and are available for re-decoding on seek.
        play = max(0, int(getattr(self, '_playhead', 0)))
        for old in list(self._prepared):
            if len(self._prepared) <= self._cache_cap:
                break
            if old < play - 2 or old > play + self._ahead + 1:
                self._prepared.pop(old, None)
        while len(self._prepared) > self._cache_cap:
            self._prepared.popitem(last=False)

    def process_new_files(self, renderer, camera):
        super().process_new_files(renderer, camera)
        renderer.qntc_live_priority_state = self
        if not self.init_ply_ok or not self.config_ok:
            self.max_play = -1
            return
        now = time.monotonic()
        if self._progressive_path.is_dir() and now >= self._scan_at:
            self._scan_at = now + 0.035
            for path in self._progressive_path.glob('NTC_*_P*.qblk'):
                if path.name in self._seen_stage_names:
                    continue
                m = _STAGENAME.fullmatch(path.name)
                if not m:
                    continue
                frame, stage = int(m.group(1)), int(m.group(2))
                with self._prep_lock:
                    self._stage_files.setdefault(frame, {})[stage] = path
                self._seen_stage_names.add(path.name)
                if stage == 3:
                    print(f'[QUIC ON DISK] NTC={frame:03d} all stages committed',flush=True)

        self._playhead = max(0, int(getattr(renderer, 'current_timestep', 0)))
        # Decode current/nearby frames only. Never expand all 299 NTCs in RAM.
        for frame in range(max(0, self._playhead - 1),
                           min(299, self._playhead + self._ahead + 1)):
            self._enqueue(frame)

        # Playback must not run ahead of DECODED contiguous NTCs, otherwise
        # renderer.query_NTC gets None and will repeatedly fail to draw.
        # Also wait until previous frame's final priority part is decoded
        # before allowing playback to advance to the following NTC.
        with self._prep_lock:
            contiguous = -1
            for f in range(min(299, self.total_frames_hint or 299)):
                if f not in self._ever_ready:
                    break
                if f > 0 and self.part_versions.get(f-1, 0) < 4:
                    break
                contiguous = f
        self.max_play = min(self.total_frames_hint - 1 if self.total_frames_hint else 299,
                            contiguous + 1) if contiguous >= 0 else -1

    def get_ntc_for_index(self, idx):
        frame = int(idx)
        with self._prep_lock:
            cached = self._prepared.get(frame)
            if cached is not None:
                self._prepared.move_to_end(frame)
        if cached is None:
            self._enqueue(frame)
            self.err = f'NTC {frame} CPU decode pending (no frame substitution)'
            return None
        version, params = cached
        if not (self.init_ply_ok and self.config_ok):
            return None
        with self._ntc_lock:
            try:
                self._ensure_ntc_single(self._renderer_ref)
                if self._progressive_loaded != (frame, version):
                    src = torch.from_numpy(params.reshape(-1))
                    dest = self._ntc_single.model.params
                    with torch.no_grad():
                        dest.copy_(src.to(device=dest.device, dtype=dest.dtype))
                    self._progressive_loaded = (frame, version)
                self.active_ntc_idx = frame
                self.active_ntc_age = 0
                return self._ntc_single
            except Exception as exc:
                self.err = f'progressive NTC load {frame}: {exc!r}'
                print('[REAL QUIC ERROR]', self.err, flush=True)
                return None

    def ui_snapshot(self):
        snap = super().ui_snapshot()
        with self._prep_lock:
            snap['ready_ntc'] = len(self._ever_ready)
            snap['ntc_files'] = sum(len(s) == 4 for s in self._stage_files.values())
            snap['max_play'] = self.max_play
            snap['progressive_cpu_cache'] = len(self._prepared)
        return snap
