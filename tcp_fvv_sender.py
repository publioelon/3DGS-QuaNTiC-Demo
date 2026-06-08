# TCP sender for QNTC/3DGStream scenes.
#
# The protocol is deliberately simple: send a relative path, send the file size,
# then send the raw file bytes. It is a practical baseline for testing
# progressive NTC delivery, not a final network protocol.
#
# The initial 3DGS scene and NTC config are sent first. The NTC files are then
# streamed in temporal order. The --ntc_stride option sends only every N-th NTC,
# which is useful for testing lower update rates and bandwidth reduction.

import argparse
import os
import socket
import struct
import time
import re
from pathlib import Path

CHUNK = 1024 * 1024

NTC_RE = re.compile(r"NTC_(\d{6})\.pth$", re.I)
ADD_RE = re.compile(r"additions_(\d{6})\.ply$", re.I)


def send_exact(sock, data):
    sock.sendall(data)


def send_file(sock, rel_path: str, abs_path: str):
    """
    Protocol:
      uint32 path_len
      path bytes
      uint64 file_size
      file bytes
    """
    rel_path = rel_path.replace("\\", "/")
    pb = rel_path.encode("utf-8")
    size = os.path.getsize(abs_path)

    send_exact(sock, struct.pack("!I", len(pb)))
    send_exact(sock, pb)
    send_exact(sock, struct.pack("!Q", size))

    t0 = time.perf_counter()
    sent = 0

    with open(abs_path, "rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            send_exact(sock, chunk)
            sent += len(chunk)

    dt = (time.perf_counter() - t0) * 1000.0
    mb = sent / (1024.0 * 1024.0)
    mbps = (sent * 8.0 / 1_000_000.0) / max(1e-6, dt / 1000.0)
    print(f"[SEND] {rel_path} ({mb:.2f} MB) in {dt:.1f} ms | {mbps:.1f} Mbps", flush=True)


def send_end(sock):
    pb = b"END"
    send_exact(sock, struct.pack("!I", len(pb)))
    send_exact(sock, pb)
    send_exact(sock, struct.pack("!Q", 0))


def wait_stable(path: Path, stable_ms=0, timeout_s=30):
    """
    Wait until a file exists and its size is stable for stable_ms.
    For prepared/offline scenes, stable_ms=0 is enough.
    For live training output, use stable_ms>0.
    """
    t0 = time.time()
    last = -1
    last_change = time.time()

    while True:
        if time.time() - t0 > timeout_s:
            return False

        if not path.exists():
            time.sleep(0.05)
            continue

        sz = path.stat().st_size
        if sz != last:
            last = sz
            last_change = time.time()

        if (time.time() - last_change) * 1000.0 >= stable_ms:
            return True

        time.sleep(0.05)


def existing_ntc_indices(ntc_dir: Path):
    out = []
    for p in sorted(ntc_dir.glob("NTC_*.pth")):
        m = NTC_RE.match(p.name)
        if m:
            out.append(int(m.group(1)))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="TCP sender for 3DGStream/QNTC scenes with optional NTC temporal stride."
    )

    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5001)

    ap.add_argument(
        "--root",
        "--src",
        dest="root",
        required=True,
        help="FVV root containing init_3dgs.ply, NTCs/, and optionally additional_3dgs/",
    )

    ap.add_argument("--start", type=int, default=0)
    ap.add_argument(
        "--end",
        type=int,
        default=-1,
        help=(
            "Last original NTC index to consider. "
            "Use -1 to infer from existing NTC files. "
            "For 299 files NTC_000000..NTC_000298, use --end 298."
        ),
    )

    ap.add_argument(
        "--ntc_stride",
        "--stride",
        dest="ntc_stride",
        type=int,
        default=1,
        help=(
            "Send only one NTC every N original indices. "
            "1=full update, 2=15 NTC/s for 30-fps source, "
            "3=10 NTC/s, 5=6 NTC/s."
        ),
    )

    ap.add_argument(
        "--poll_ms",
        type=int,
        default=50,
        help="Polling interval when waiting for live-generated files.",
    )

    ap.add_argument(
        "--stable_ms",
        type=int,
        default=0,
        help="Require file size to stay stable for this many ms before sending.",
    )

    ap.add_argument(
        "--no_additions",
        action="store_true",
        help="Do not send additional_3dgs/additions_*.ply even if present.",
    )

    ap.add_argument(
        "--connect_timeout",
        type=float,
        default=10.0,
        help="TCP connection timeout in seconds.",
    )

    args = ap.parse_args()

    args.ntc_stride = max(1, int(args.ntc_stride))

    root = Path(args.root)
    init_ply = root / "init_3dgs.ply"
    ntc_dir = root / "NTCs"
    cfg = ntc_dir / "config.json"

    add_dir = root / "additional_3dgs"
    if not add_dir.exists():
        alt = root / "additional_3dgs_OFF"
        if alt.exists():
            add_dir = alt

    if not wait_stable(init_ply, stable_ms=args.stable_ms, timeout_s=30):
        raise FileNotFoundError(init_ply)
    if not wait_stable(cfg, stable_ms=args.stable_ms, timeout_s=30):
        raise FileNotFoundError(cfg)

    available = existing_ntc_indices(ntc_dir)
    if not available and args.end < 0:
        raise FileNotFoundError(f"No NTC_*.pth files found in {ntc_dir}")

    if args.end >= 0:
        end_idx = int(args.end)
    else:
        end_idx = max(available)

    selected_indices = [
        i for i in range(int(args.start), end_idx + 1)
        if ((i - int(args.start)) % args.ntc_stride) == 0
    ]

    print("[SENDER] root:", root, flush=True)
    print("[SENDER] start/end:", args.start, end_idx, flush=True)
    print("[SENDER] ntc_stride:", args.ntc_stride, flush=True)
    print("[SENDER] selected NTC count:", len(selected_indices), flush=True)
    if selected_indices:
        print("[SENDER] first/last selected:", selected_indices[0], selected_indices[-1], flush=True)

    with socket.create_connection((args.host, args.port), timeout=args.connect_timeout) as s:
        try:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass

        print(f"[SENDER] connected to {args.host}:{args.port}", flush=True)

        # Always send the base scene and config first.
        send_file(s, "init_3dgs.ply", str(init_ply))
        send_file(s, "NTCs/config.json", str(cfg))

        sent_ntc = set()
        sent_add = set()

        for i in selected_indices:
            ntc_path = ntc_dir / f"NTC_{i:06d}.pth"
            add_path = add_dir / f"additions_{i:06d}.ply"

            # Send NTC as soon as it exists and is stable.
            while i not in sent_ntc:
                if ntc_path.exists() and wait_stable(
                    ntc_path,
                    stable_ms=args.stable_ms,
                    timeout_s=300,
                ):
                    send_file(s, f"NTCs/{ntc_path.name}", str(ntc_path))
                    sent_ntc.add(i)
                    break

                print(f"[SENDER] waiting for {ntc_path}", flush=True)
                time.sleep(args.poll_ms / 1000.0)

            # Optional additions. For stable no-additions QNTC scenes this is skipped.
            if (not args.no_additions) and add_dir.exists() and add_path.exists():
                while i not in sent_add:
                    if add_path.exists() and wait_stable(
                        add_path,
                        stable_ms=args.stable_ms,
                        timeout_s=300,
                    ):
                        send_file(s, f"{add_dir.name}/{add_path.name}", str(add_path))
                        sent_add.add(i)
                        break

                    print(f"[SENDER] waiting for {add_path}", flush=True)
                    time.sleep(args.poll_ms / 1000.0)

        send_end(s)

    print("[SENDER] done", flush=True)


if __name__ == "__main__":
    main()
