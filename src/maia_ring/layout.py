"""The ring's byte layout. docs/ring-format.md describes the same thing for other languages.

    [ header page ][ slot table ][ slot 0 data ][ slot 1 data ] ...

All integers are little-endian. Every region starts on a 4096-byte page so a slot can be
registered as-is with Triton's system shared memory (or mapped by another process).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

MAGIC = b"MAIARING"
VERSION = 1
PAGE = 4096

# header page: fixed fields, then the commit counter on its own cache line
HEADER = struct.Struct("<8sIIIIIIIIII")  # magic, version, header_size, table_offset, slot_count,
#                                           slot_stride, data_offset, width, height, channels, pixfmt
COMMIT_OFFSET = 64  # u64: index of the newest committed frame + 1 (0 = nothing written yet)
COMMIT = struct.Struct("<Q")
INSTANCE_OFFSET = 72  # u64: random id of the writer that created the segment (changes on restart)
INSTANCE = struct.Struct("<Q")

# one 64-byte entry per slot in the slot table
SLOT = struct.Struct("<QQQQIIIIII8x")  # seq, frame_index, timestamp_ns, pts_ns, flags,
#                                         width, height, channels, pixfmt, nbytes (+ padding)
SLOT_SIZE = 64
assert SLOT.size == SLOT_SIZE

PIXFMTS = {"bgr": 1, "rgb": 2, "gray": 3, "nv12": 4, "bgra": 5}
PIXFMT_NAMES = {v: k for k, v in PIXFMTS.items()}

# frame flags
IFRAME = 1  # decoded from a key frame
MOTION = 2  # passed the motion gate
DISCONTINUITY = 4  # first frame after a reconnect or a profile change


def align(n: int, to: int = PAGE) -> int:
    return (n + to - 1) // to * to


def frame_shape(width: int, height: int, channels: int, pixfmt: str) -> tuple[int, ...]:
    if pixfmt == "nv12":  # Y plane + interleaved UV at half height, as one 2-D array
        return (height * 3 // 2, width)
    return (height, width) if channels == 1 else (height, width, channels)


@dataclass(frozen=True)
class Geometry:
    slot_count: int
    width: int
    height: int
    channels: int
    pixfmt: str

    @property
    def shape(self) -> tuple[int, ...]:
        return frame_shape(self.width, self.height, self.channels, self.pixfmt)

    @property
    def frame_bytes(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def table_offset(self) -> int:
        return PAGE

    @property
    def data_offset(self) -> int:
        return PAGE + align(self.slot_count * SLOT_SIZE)

    @property
    def slot_stride(self) -> int:
        return align(self.frame_bytes)

    @property
    def total_bytes(self) -> int:
        return self.data_offset + self.slot_count * self.slot_stride

    def slot_offset(self, slot: int) -> int:
        return self.data_offset + slot * self.slot_stride

    def entry_offset(self, slot: int) -> int:
        return self.table_offset + slot * SLOT_SIZE

    def pack_header(self) -> bytes:
        return HEADER.pack(
            MAGIC, VERSION, PAGE, self.table_offset, self.slot_count, self.slot_stride, self.data_offset,
            self.width, self.height, self.channels, PIXFMTS[self.pixfmt],
        )  # fmt: skip

    @classmethod
    def unpack_header(cls, buf) -> Geometry:
        magic, version, _, table, count, stride, data, w, h, c, fmt = HEADER.unpack_from(buf, 0)
        if magic != MAGIC:
            raise ValueError("not a maia-ring segment (bad magic)")
        if version != VERSION:
            raise ValueError(f"ring format version {version}, this maia-ring reads version {VERSION}")
        g = cls(count, w, h, c, PIXFMT_NAMES[fmt])
        if (table, data, stride) != (g.table_offset, g.data_offset, g.slot_stride):
            raise ValueError("ring header is inconsistent with its geometry")
        return g
