# Changelog

## 0.1.3
- `RingWriter(..., group=gid)` (or `MAIA_RING_GID` in the writer's environment) lets readers
  in that group open the ring (mode 0660). Rings stay private (0600) by default. Needed when
  the writer runs as another user, e.g. maia-gateway in Docker (root) with readers on the host.
- A reader denied access now says why and what to set, instead of a bare `PermissionError`.

## 0.1.2
- Rings no longer use Python's resource tracker at all. Its bookkeeping, shared across
  processes, could print "leaked shared_memory" warnings at exit (seen on Python 3.11).
  The writer now deletes its ring on `close()` and at normal interpreter exit, unless a
  newer writer has replaced it; a ring left by a crashed writer is replaced by the next
  writer or deleted with the new `maia-ring rm`.

## 0.1.1
- Readers never register with Python's resource tracker (before 3.13). Registering then
  unregistering also dropped the writer's registration when both lived in one process, so the
  tracker printed a KeyError on close and could not clean up after a crashed writer.
- Apple Silicon numbers in the README.

## 0.1.0
- First release: single-writer, many-reader frame ring in POSIX shared memory with a
  per-slot seqlock; copy or zero-copy reads; `FrameRef` to pass frames between processes by
  reference; `RingPool` for readers serving many rings; writer-restart detection; page-aligned
  slots (Triton system shared memory); `maia-ring bench` and `maia-ring info`; format spec.
