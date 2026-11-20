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


def attach(name: str) -> shared_memory.SharedMemory:
    """Attach without registering the segment with this process's resource tracker.
    Before Python 3.13 the tracker unlinks every segment a process *attached to* when that
    process exits, so a reader quitting would delete the writer's ring."""
    if sys.version_info >= (3, 13):
        return shared_memory.SharedMemory(name=name, track=False)
    shm = shared_memory.SharedMemory(name=name)
    try:
        resource_tracker.unregister(shm._name, "shared_memory")  # noqa: SLF001
    except Exception:  # tracker not running (e.g. under some process launchers)
        pass
    return shm
