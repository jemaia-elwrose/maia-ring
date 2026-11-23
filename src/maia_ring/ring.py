"""Writer and reader for a single-writer, many-reader frame ring in shared memory.

Each slot is guarded by a seqlock: the writer makes the slot's sequence number odd, writes
the frame, then makes it even again. A reader notes the number, reads, and checks it again;
a change means the frame was overwritten while being read, so the read is retried or
reported as stale. No locks are shared between processes, so a slow or crashed reader can
never block the writer.
"""

from __future__ import annotations

import atexit
import os
import random
import time
from dataclasses import asdict, dataclass

import numpy as np

from . import _shm
from . import layout as L


class StaleFrame(Exception):
    """The frame was overwritten (or the writer restarted) before it could be read."""


@dataclass(frozen=True)
class FrameRef:
    """Everything another process needs to find one frame. Small and JSON-safe, so it can
    travel in a request instead of the pixels."""

    ring: str
    instance: int
    slot: int
    frame_index: int
    seq: int
    shape: tuple[int, ...]
    pixfmt: str
    timestamp_ns: int
    dtype: str = "uint8"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["shape"] = list(self.shape)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> FrameRef:
        return cls(**{**d, "shape": tuple(d["shape"])})


@dataclass(frozen=True)
class FrameMeta:
    frame_index: int
    timestamp_ns: int
    pts_ns: int
    flags: int

    @property
    def is_iframe(self) -> bool:
        return bool(self.flags & L.IFRAME)


class Frame:
    """A frame read from the ring. ``array`` is a copy unless the read asked for a view; a
    view points straight into shared memory, so call ``still_valid()`` after using it."""

    __slots__ = ("array", "meta", "ref", "_reader", "_copied")

    def __init__(self, array: np.ndarray, meta: FrameMeta, ref: FrameRef, reader: RingReader, copied: bool):
        self.array, self.meta, self.ref, self._reader, self._copied = array, meta, ref, reader, copied

    def still_valid(self) -> bool:
        return self._copied or self._reader._seq(self.ref.slot) == self.ref.seq


def _random_instance() -> int:
    return random.SystemRandom().getrandbits(63) or 1


def _remove_if_ours(name: str, instance: int) -> None:
    """At interpreter exit: delete the ring unless a newer writer has replaced it."""
    try:
        shm = _shm.attach(name)
    except FileNotFoundError:
        return
    try:
        ours = L.INSTANCE.unpack_from(shm.buf, L.INSTANCE_OFFSET)[0] == instance
    finally:
        shm.close()
    if ours:
        _shm.remove(name)


_OPEN_WRITERS: dict[int, tuple[str, int]] = {}  # id(writer) -> (ring name, instance)


@atexit.register
def _cleanup_open_writers() -> None:
    for name, instance in list(_OPEN_WRITERS.values()):
        _remove_if_ours(name, instance)


class RingWriter:
    """Owns the segment. One writer per ring."""

    def __init__(
        self,
        name: str,
        *,
        slots: int,
        width: int,
        height: int,
        channels: int = 3,
        pixfmt: str = "bgr",
        group: int | None = None,
    ):
        """``group``: a group id allowed to read the ring (mode 0660). By default only the
        writer's user can open it (0600), which stops readers running as another user, for
        example a writer in Docker (root) and readers on the host. ``MAIA_RING_GID`` in the
        environment sets the default."""
        if slots < 2:
            raise ValueError("a ring needs at least 2 slots")
        if pixfmt not in L.PIXFMTS:
            raise ValueError(f"pixfmt must be one of {sorted(L.PIXFMTS)}")
        self.name = _shm.check_name(name)
        self.geometry = g = L.Geometry(slots, width, height, channels, pixfmt)
        self._shm = _shm.create(name, g.total_bytes)
        if group is None and os.environ.get("MAIA_RING_GID", "").strip():
            group = int(os.environ["MAIA_RING_GID"])
        if group is not None:
            _shm.share_with_group(self._shm, group)
        self._buf = self._shm.buf
        self._buf[: L.PAGE] = bytes(L.PAGE)
        self._buf[g.table_offset : g.data_offset] = bytes(g.data_offset - g.table_offset)
        self.instance = _random_instance()
        L.INSTANCE.pack_into(self._buf, L.INSTANCE_OFFSET, self.instance)
        L.COMMIT.pack_into(self._buf, L.COMMIT_OFFSET, 0)
        self._buf[: L.HEADER.size] = g.pack_header()  # magic last-ish: readers see a complete header
        self._next = 0
        self._open: tuple[int, int, int] | None = None  # (slot, frame_index, seq) being written
        self._views = [np.ndarray(g.shape, np.uint8, self._buf, g.slot_offset(i)) for i in range(slots)]
        _OPEN_WRITERS[id(self)] = (name, self.instance)

    def begin(self) -> np.ndarray:
        """Reserve the next slot and return a writable view of it, for writing a frame in
        place (for example straight from a decoder buffer). Finish with :meth:`commit`."""
        if self._open is not None:
            raise RuntimeError("commit() the previous frame first")
        g = self.geometry
        index = self._next
        slot = index % g.slot_count
        entry = g.entry_offset(slot)
        seq = L.COMMIT.unpack_from(self._buf, entry)[0] + 1  # odd: being written
        L.COMMIT.pack_into(self._buf, entry, seq)
        _shm.fence()
        self._open = (slot, index, seq)
        return self._views[slot]

    def commit(self, *, timestamp_ns: int | None = None, pts_ns: int = 0, flags: int = 0) -> FrameRef:
        if self._open is None:
            raise RuntimeError("begin() a frame first")
        g = self.geometry
        slot, index, seq = self._open
        ts = time.time_ns() if timestamp_ns is None else timestamp_ns
        entry = g.entry_offset(slot)
        L.SLOT.pack_into(
            self._buf, entry, seq, index, ts, pts_ns, flags, g.width, g.height, g.channels,
            L.PIXFMTS[g.pixfmt], g.frame_bytes,
        )  # fmt: skip
        _shm.fence()
        L.COMMIT.pack_into(self._buf, entry, seq + 1)  # even: complete
        _shm.fence()
        L.COMMIT.pack_into(self._buf, L.COMMIT_OFFSET, index + 1)
        self._open = None
        self._next = index + 1
        return FrameRef(self.name, self.instance, slot, index, seq + 1, g.shape, g.pixfmt, ts)

    def write(self, frame: np.ndarray, **meta) -> FrameRef:
        """Copy one frame into the ring (one memcpy) and publish it."""
        view = self.begin()
        try:
            np.copyto(view, frame.reshape(view.shape), casting="no")
        except Exception:
            self._abort()
            raise
        return self.commit(**meta)

    def _abort(self) -> None:
        slot, _, seq = self._open
        entry = self.geometry.entry_offset(slot)
        L.COMMIT.pack_into(self._buf, entry + 8, 2**64 - 1)  # frame_index no reader will ask for
        _shm.fence()
        L.COMMIT.pack_into(self._buf, entry, seq + 1)  # even again; the half-written data is unreachable
        self._open = None

    def close(self, unlink: bool = True) -> None:
        self._views = []
        self._buf = None
        try:
            self._shm.close()
        except BufferError as e:
            raise BufferError("close() while a view from begin() is still alive; delete it first") from e
        _OPEN_WRITERS.pop(id(self), None)
        if unlink:
            _shm.unlink(self._shm)

    def __enter__(self) -> RingWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class RingReader:
    """Attaches to an existing ring. Any number of readers, in any number of processes."""

    def __init__(self, name: str):
        self.name = _shm.check_name(name)
        self._shm = _shm.attach(name)
        self._buf = self._shm.buf
        self.geometry = g = L.Geometry.unpack_header(self._buf)
        self.instance = L.INSTANCE.unpack_from(self._buf, L.INSTANCE_OFFSET)[0]
        self._views = [np.ndarray(g.shape, np.uint8, self._buf, g.slot_offset(i)) for i in range(g.slot_count)]
        for v in self._views:
            v.flags.writeable = False

    # -- low level --
    def _seq(self, slot: int) -> int:
        return L.COMMIT.unpack_from(self._buf, self.geometry.entry_offset(slot))[0]

    def committed(self) -> int:
        """Number of frames written so far (the newest has index ``committed() - 1``)."""
        return L.COMMIT.unpack_from(self._buf, L.COMMIT_OFFSET)[0]

    def replaced(self) -> bool:
        """True if the writer restarted and made a new segment under this name."""
        try:
            fresh = _shm.attach(self.name)
        except FileNotFoundError:
            return True
        try:
            return L.INSTANCE.unpack_from(fresh.buf, L.INSTANCE_OFFSET)[0] != self.instance
        finally:
            fresh.close()

    def _read(self, slot: int, index: int | None, copy: bool) -> Frame:
        g = self.geometry
        s1 = self._seq(slot)
        if s1 % 2 or s1 == 0:
            raise StaleFrame("slot is being written" if s1 else "slot never written")
        _shm.fence()
        _, fi, ts, pts, flags, *_ = L.SLOT.unpack_from(self._buf, g.entry_offset(slot))
        if index is not None and fi != index:
            raise StaleFrame(f"slot {slot} now holds frame {fi}, not {index}")
        view = self._views[slot]
        array = view.copy() if copy else view
        _shm.fence()
        if self._seq(slot) != s1:
            raise StaleFrame(f"frame {fi} was overwritten while being read")
        ref = FrameRef(self.name, self.instance, slot, fi, s1, g.shape, g.pixfmt, ts)
        return Frame(array, FrameMeta(fi, ts, pts, flags), ref, self, copy)

    # -- public --
    def latest(self, copy: bool = True) -> Frame | None:
        """The newest complete frame, or None if nothing has been written yet."""
        for _ in range(4):
            n = self.committed()
            if n == 0:
                return None
            try:
                return self._read((n - 1) % self.geometry.slot_count, n - 1, copy)
            except StaleFrame:
                continue  # overwritten under us: a newer frame exists, take that
        raise StaleFrame("writer is lapping the reader; use more slots")

    def get(self, ref: FrameRef, copy: bool = True) -> Frame:
        """A specific frame, if it is still in the ring."""
        if ref.instance != self.instance:
            raise StaleFrame("the ring was recreated by a new writer")
        frame = self._read(ref.slot, ref.frame_index, copy)
        if frame.ref.seq != ref.seq:
            raise StaleFrame(f"frame {ref.frame_index} was overwritten")
        return frame

    def wait(self, after: int = -1, timeout: float | None = None, poll: float = 0.001) -> Frame | None:
        """Block until a frame newer than index ``after`` exists, then return the newest one."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.committed() - 1 <= after:
            if deadline is not None and time.monotonic() >= deadline:
                return None
            time.sleep(poll)
        return self.latest()

    def close(self) -> None:
        """Drop every view first: shared memory cannot be unmapped while numpy arrays point into it."""
        self._views = []
        self._buf = None
        try:
            self._shm.close()
        except BufferError as e:
            raise BufferError("close() while views from this reader are still alive; delete them first") from e

    def __enter__(self) -> RingReader:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class RingPool:
    """Readers cached by ring name, for a process that serves frames from many cameras
    (a maia-infer worker). Reattaches by itself when a writer restarts."""

    def __init__(self) -> None:
        self._readers: dict[str, RingReader] = {}

    def reader(self, name: str) -> RingReader:
        r = self._readers.get(name)
        if r is None:
            r = self._readers[name] = RingReader(name)
        return r

    def get(self, ref: FrameRef | dict, copy: bool = False) -> Frame:
        ref = FrameRef.from_dict(ref) if isinstance(ref, dict) else ref
        r = self.reader(ref.ring)
        if ref.instance != r.instance:  # writer restarted since we attached
            self._drop(ref.ring)
            r = self.reader(ref.ring)
        return r.get(ref, copy=copy)

    def _drop(self, name: str) -> None:
        r = self._readers.pop(name, None)
        if r is not None:
            try:
                r.close()
            except BufferError:
                pass  # views still held somewhere; the mapping is freed when they go

    def close(self) -> None:
        for name in list(self._readers):
            self._drop(name)
