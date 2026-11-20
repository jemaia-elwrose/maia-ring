# maia-ring format, version 1

Everything a reader in another language needs to read frames from a maia-ring segment.
All integers are little-endian. One segment per ring, a POSIX shared-memory object named
`/<ring name>` (1–30 characters, no `/`; macOS caps names at 31 bytes).

```
offset 0                 4096                       data_offset
| header page (4096 B)   | slot table (64 B/slot,   | slot 0 | slot 1 | ... |
|                        |  padded to a page)       | each slot_stride bytes, page-aligned
```

## Header page

| Offset | Type | Field |
|---|---|---|
| 0 | 8 bytes | magic `MAIARING` |
| 8 | u32 | version (1) |
| 12 | u32 | header size (4096) |
| 16 | u32 | slot table offset (4096) |
| 20 | u32 | slot count |
| 24 | u32 | slot stride (frame bytes rounded up to 4096) |
| 28 | u32 | data offset (4096 + slot table rounded up to 4096) |
| 32 | u32 | width |
| 36 | u32 | height |
| 40 | u32 | channels |
| 44 | u32 | pixel format: 1 bgr, 2 rgb, 3 gray, 4 nv12, 5 bgra |
| 64 | u64 | **commit counter**: frames written so far; the newest is `counter - 1` |
| 72 | u64 | **instance**: random id of the writer; changes when a writer recreates the ring |

Frame index `i` lives in slot `i % slot_count`, at `data_offset + slot * slot_stride`.
A frame is uint8, `height × width × channels` (gray: `height × width`; nv12: `height·3/2 × width`).

## Slot table entry (64 bytes)

| Offset | Type | Field |
|---|---|---|
| 0 | u64 | **seq**: odd while the slot is being written, even when complete, 0 if never written |
| 8 | u64 | frame index |
| 16 | u64 | capture timestamp, ns since the Unix epoch |
| 24 | u64 | presentation timestamp from the stream, ns |
| 32 | u32 | flags: 1 I-frame, 2 passed motion gate, 4 discontinuity (reconnect or profile change) |
| 36 | u32 | width |
| 40 | u32 | height |
| 44 | u32 | channels |
| 48 | u32 | pixel format |
| 52 | u32 | frame bytes |

## Writing (one writer per ring)

1. `seq = seq + 1` (now odd), then a memory barrier.
2. Write the frame data and the entry's other fields.
3. Barrier, `seq = seq + 1` (even), barrier.
4. `commit counter = frame index + 1`.

## Reading (any number of readers, no locks)

1. `s1 = seq`. If `s1` is odd or 0, the slot is not readable now.
2. Barrier. Read the entry; if a specific frame was asked for, check its frame index.
3. Read or use the frame data.
4. Barrier. If `seq != s1`, the writer reused the slot during the read: discard the data.

A reader that hands out a pointer into the slot (no copy) must do step 4 *after* the
consumer is finished with the data, and discard whatever it produced if the check fails.

To read the newest frame: `n = commit counter`, slot `(n - 1) % slot_count`, expecting frame
index `n - 1`; if the check fails, read the counter again and retry.

## Writer restarts

A restarted writer replaces the segment and picks a new instance id. A reader keeps the old
mapping until it reattaches; it notices the change by opening the name again and comparing
the instance field, or when a request carries an instance it does not have.

## Frame references

A frame can travel between processes as a small reference instead of its pixels:

```json
{"ring": "cam-gate-1", "instance": 7312…, "slot": 3, "frame_index": 1187, "seq": 298,
 "shape": [720, 1280, 3], "pixfmt": "bgr", "timestamp_ns": 1795…, "dtype": "uint8"}
```

The receiver opens the ring, checks `instance`, then reads slot `slot` expecting `frame_index`
and `seq`. Any mismatch means the frame is gone: ask for a newer one.
