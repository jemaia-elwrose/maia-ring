"""Shared-memory plumbing that the standard library gets subtly wrong for this use."""

from __future__ import annotations

import sys
import threading
from multiprocessing import resource_tracker, shared_memory

_fence_lock = threading.Lock()


def fence() -> None:
    """Best-effort memory barrier. Python has no fence primitive; locking and unlocking a
    mutex issues acquire and release barriers in C. On x86 stores are never reordered with
    other stores, so the seqlock is exact there. On ARM this is a practical barrier, not a
    formal full fence; a C extension with __atomic_thread_fence is the planned upgrade."""
    _fence_lock.acquire()
    _fence_lock.release()


def check_name(name: str) -> str:
    # macOS limits POSIX shared-memory names to 31 bytes including the leading slash
    if not name or "/" in name or len(name) > 30:
        raise ValueError(f"ring name must be 1-30 characters without '/': {name!r}")
    return name


def create(name: str, size: int) -> shared_memory.SharedMemory:
    """Create a segment, replacing a stale one left behind by a crashed writer."""
    try:
        return shared_memory.SharedMemory(name=name, create=True, size=size)
    except FileExistsError:
        stale = attach(name)
        stale.close()
        stale.unlink()
        return shared_memory.SharedMemory(name=name, create=True, size=size)


_register_lock = threading.Lock()


def attach(name: str) -> shared_memory.SharedMemory:
    """Attach without registering the segment with this process's resource tracker.

    Before Python 3.13 attaching registers the segment, and the tracker unlinks every
    registered segment when the process exits, so a reader quitting would delete the
    writer's ring. Unregistering afterwards is not enough: when the writer lives in the
    same process, that also drops the writer's own registration. So registration is
    skipped while attaching instead."""
    if sys.version_info >= (3, 13):
        return shared_memory.SharedMemory(name=name, track=False)
    with _register_lock:
        original = resource_tracker.register
        resource_tracker.register = lambda *args, **kwargs: None
        try:
            return shared_memory.SharedMemory(name=name)
        finally:
            resource_tracker.register = original
