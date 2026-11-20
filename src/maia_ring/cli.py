"""maia-ring command line.

maia-ring bench                 # cost of writing and reading frames on this machine
maia-ring info cam-gate-1       # header and state of a live ring
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time

import numpy as np

from .ring import RingReader, RingWriter

SIZES = {"720p": (1280, 720), "1080p": (1920, 1080), "4k": (3840, 2160)}


def _per_frame_us(fn, n: int) -> float:
    fn()  # warm up
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6


def bench_size(size: str, n: int) -> dict:
    width, height = SIZES[size]
    name = f"bench{os.getpid() % 100000}"
    src = np.random.default_rng(0).integers(0, 255, (height, width, 3), np.uint8)
    with RingWriter(name, slots=8, width=width, height=height) as w, RingReader(name) as r:
        write_us = _per_frame_us(lambda: w.write(src), n)
        read_copy_us = _per_frame_us(lambda: r.latest(copy=True), n)
        ref = w.write(src)  # the writer is idle from here, so this frame stays readable

        def view_read():
            r.get(ref, copy=False).still_valid()

        row = {
            "size": size,
            "MB": round(src.nbytes / 1e6, 2),
            "write_us": write_us,
            "read_copy_us": read_copy_us,
            "read_view_us": _per_frame_us(view_read, n),
            "numpy_copy_us": _per_frame_us(src.copy, n),  # baseline: a plain memcpy of one frame
        }
        del view_read
    return row


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="maia-ring", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="measure write and read cost per frame")
    b.add_argument("--sizes", nargs="+", default=["720p", "1080p"], choices=sorted(SIZES))
    b.add_argument("-n", type=int, default=300, help="frames per measurement")
    b.add_argument("--json", action="store_true")
    i = sub.add_parser("info", help="show a live ring's header and newest frame")
    i.add_argument("name")
    args = ap.parse_args(argv)

    if args.cmd == "bench":
        rows = [bench_size(size, args.n) for size in args.sizes]
        if args.json:
            print(json.dumps({"machine": platform.platform(), "rows": rows}, indent=2))
            return
        print(f"{platform.platform()}  ({args.n} frames each, BGR uint8, microseconds per frame)")
        print(f"{'size':<7}{'MB':>6}{'write':>10}{'read copy':>12}{'read view':>12}{'memcpy':>10}")
        for r in rows:
            print(
                f"{r['size']:<7}{r['MB']:>6}{r['write_us']:>10.0f}{r['read_copy_us']:>12.0f}"
                f"{r['read_view_us']:>12.1f}{r['numpy_copy_us']:>10.0f}"
            )
        return

    with RingReader(args.name) as r:
        g, n = r.geometry, r.committed()
        f = r.latest()
        age = (time.time_ns() - f.meta.timestamp_ns) / 1e6 if f else None
        del f
    info = {
        "ring": args.name,
        "slots": g.slot_count,
        "size": f"{g.width}x{g.height}",
        "pixfmt": g.pixfmt,
        "frame_MB": round(g.frame_bytes / 1e6, 2),
        "total_MB": round(g.total_bytes / 1e6, 1),
        "frames_written": n,
        "newest_age_ms": None if age is None else round(age, 1),
    }
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
