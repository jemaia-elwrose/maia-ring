"""Shared-memory plumbing that the standard library gets subtly wrong for this use.

maia-ring never uses Python's multiprocessing resource tracker. The tracker deletes a
segment when *any* process that touched it exits (before Python 3.13 that includes readers),
and its bookkeeping, shared by every process it serves, can disagree with what really exists
(KeyErrors, "leaked shared_memory" warnings). Rings manage their own lifetime instead: the
writer deletes its ring on close and at interpreter exit, and a new writer replaces a ring
left behind by one that crashed.
"""

from __future__ import annotations

import contextlib
import sys
import threading
from multiprocessing import resource_tracker, shared_memory

_fence_lock = threading.Lock()
_tracker_lock = threading.Lock()


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


@contextlib.contextmanager
def _no_tracker():
    """Before Python 3.13 SharedMemory always talks to the resource tracker; silence it."""
    with _tracker_lock:
        register, unregister = resource_tracker.register, resource_tracker.unregister
        resource_tracker.register = resource_tracker.unregister = lambda *args, **kwargs: None
        try:
            yield
        finally:
            resource_tracker.register, resource_tracker.unregister = register, unregister


def _open(name: str, create: bool = False, size: int = 0) -> shared_memory.SharedMemory:
    if sys.version_info >= (3, 13):
        return shared_memory.SharedMemory(name=name, create=create, size=size, track=False)
    with _no_tracker():
        return shared_memory.SharedMemory(name=name, create=create, size=size)


def attach(name: str) -> shared_memory.SharedMemory:
    return _open(name)


def unlink(shm: shared_memory.SharedMemory) -> None:
    with _no_tracker():  # on 3.13 an untracked segment's unlink() skips the tracker already
        try:
            shm.unlink()
        except FileNotFoundError:
            pass


def create(name: str, size: int) -> shared_memory.SharedMemory:
    """Create a segment, replacing a stale one left behind by a crashed writer."""
    try:
        return _open(name, create=True, size=size)
    except FileExistsError:
        stale = attach(name)
        stale.close()
        unlink(stale)
        return _open(name, create=True, size=size)


def remove(name: str) -> bool:
    """Delete a ring by name (e.g. one left by a crashed writer). False if there was none."""
    try:
        shm = attach(name)
    except FileNotFoundError:
        return False
    shm.close()
    unlink(shm)
    return True
