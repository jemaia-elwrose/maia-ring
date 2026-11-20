# Changelog

## 0.1.0
- First release: single-writer, many-reader frame ring in POSIX shared memory with a
  per-slot seqlock; copy or zero-copy reads; `FrameRef` to pass frames between processes by
  reference; `RingPool` for readers serving many rings; writer-restart detection; page-aligned
  slots (Triton system shared memory); `maia-ring bench` and `maia-ring info`; format spec. Readers never register with Python's
  resource tracker (before 3.13), so a reader in the writer's own process can't break its cleanup.
