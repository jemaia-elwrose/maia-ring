import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pytest

from maia_ring import IFRAME, FrameRef, Geometry, RingPool, RingReader, RingWriter, StaleFrame
from maia_ring import layout as L


@pytest.fixture
def name(request):
    # short and unique: macOS allows 30 characters
    return f"t{os.getpid() % 100000}{abs(hash(request.node.name)) % 100000}"


def frame(value, h=48, w=64, c=3):
    return np.full((h, w, c), value % 256, np.uint8)


def test_write_then_read_latest(name):
    with RingWriter(name, slots=4, width=64, height=48) as w, RingReader(name) as r:
        assert r.latest() is None
        w.write(frame(5), flags=IFRAME, pts_ns=33)
        f = r.latest()
        assert f.array.shape == (48, 64, 3) and f.array.mean() == 5
        assert f.meta.frame_index == 0 and f.meta.is_iframe and f.meta.pts_ns == 33
        assert r.geometry == Geometry(4, 64, 48, 3, "bgr")


def test_read_by_ref_and_stale_after_lapping(name):
    with RingWriter(name, slots=3, width=64, height=48) as w, RingReader(name) as r:
        ref = w.write(frame(1))
        assert r.get(ref).array.mean() == 1
        for i in range(3):
            w.write(frame(10 + i))
        with pytest.raises(StaleFrame):
            r.get(ref)
        assert r.latest().meta.frame_index == 3


def test_view_reports_when_overwritten(name):
    with RingWriter(name, slots=2, width=64, height=48) as w, RingReader(name) as r:
        ref = w.write(frame(1))
        view = r.get(ref, copy=False)
        assert not view.array.flags.writeable and view.still_valid()
        w.write(frame(2))
        assert view.still_valid()  # other slot
        w.write(frame(3))
        assert not view.still_valid()  # its slot was reused
        del view


def test_ref_round_trips_through_json(name):
    with RingWriter(name, slots=2, width=64, height=48) as w, RingReader(name) as r:
        ref = w.write(frame(9))
        back = FrameRef.from_dict(json.loads(json.dumps(ref.to_dict())))
        assert back == ref and r.get(back).array.mean() == 9


def test_begin_commit_in_place_and_abort(name):
    with RingWriter(name, slots=2, width=64, height=48) as w, RingReader(name) as r:
        view = w.begin()
        view[:] = 4
        del view
        ref = w.commit(timestamp_ns=123)
        assert r.get(ref).array.mean() == 4 and ref.timestamp_ns == 123
        with pytest.raises(ValueError):
            w.write(np.zeros((10, 10, 3), np.uint8))  # wrong size: aborted, nothing published
        assert r.committed() == 1 and r.latest().meta.frame_index == 0
        with pytest.raises(RuntimeError):
            w.commit()


@pytest.mark.parametrize("fmt,channels,shape", [("gray", 1, (48, 64)), ("nv12", 1, (72, 64)), ("bgra", 4, (48, 64, 4))])
def test_pixel_formats(name, fmt, channels, shape):
    with RingWriter(name, slots=2, width=64, height=48, channels=channels, pixfmt=fmt) as w, RingReader(name) as r:
        w.write(np.ones(shape, np.uint8))
        assert r.latest().array.shape == shape and r.geometry.pixfmt == fmt


def test_slots_are_page_aligned_for_triton():
    g = Geometry(5, 1920, 1080, 3, "bgr")
    assert all(g.slot_offset(i) % L.PAGE == 0 for i in range(5))
    assert g.total_bytes == g.slot_offset(4) + g.slot_stride


def test_rejects_bad_names_and_foreign_segments(name):
    with pytest.raises(ValueError):
        RingWriter("x" * 31, slots=2, width=8, height=8)
    from multiprocessing import shared_memory

    other = shared_memory.SharedMemory(name=name, create=True, size=8192)
    try:
        with pytest.raises(ValueError, match="magic"):
            RingReader(name)
    finally:
        other.close()
        other.unlink()


def test_wait_times_out_then_returns_new_frame(name):
    with RingWriter(name, slots=2, width=64, height=48) as w, RingReader(name) as r:
        assert r.wait(timeout=0.02) is None
        w.write(frame(1))
        assert r.wait(after=-1, timeout=1).meta.frame_index == 0


def test_writer_restart_is_detected_and_pool_reattaches(name):
    pool = RingPool()
    w = RingWriter(name, slots=2, width=64, height=48)
    old = w.write(frame(1))
    assert pool.get(old, copy=True).array.mean() == 1
    w.close()
    w = RingWriter(name, slots=2, width=64, height=48)  # e.g. ingest process restarted
    new = w.write(frame(2))
    assert pool.reader(name).replaced()
    assert pool.get(new.to_dict(), copy=True).array.mean() == 2
    with pytest.raises(StaleFrame):
        pool.get(old)
    pool.close()
    w.close()


def _attach_and_exit(name):
    RingReader(name).close()


def test_reader_process_exit_does_not_delete_the_ring(name):
    # before Python 3.13 the resource tracker unlinked segments that a process merely attached to
    with RingWriter(name, slots=2, width=64, height=48) as w:
        w.write(frame(1))
        p = mp.get_context("spawn").Process(target=_attach_and_exit, args=(name,))
        p.start()
        p.join(30)
        assert p.exitcode == 0
        time.sleep(0.2)
        with RingReader(name) as r:
            assert r.latest().array.mean() == 1


def _writer(name, seconds, ready, count):
    with RingWriter(name, slots=3, width=320, height=240) as w:
        ready.set()
        deadline, i = time.monotonic() + seconds, 0
        while time.monotonic() < deadline:
            view = w.begin()
            view[:] = i % 256  # a frame is uniform: any torn read shows two values
            del view
            w.commit()
            i += 1
        count.value = i


def _reader(name, seconds, use_views, out):
    deadline = time.monotonic() + seconds
    ok = stale = torn = 0
    with RingReader(name) as r:
        while time.monotonic() < deadline:
            try:
                f = r.latest(copy=not use_views)
            except StaleFrame:
                stale += 1
                continue
            if f is None:
                continue
            a = f.array
            lo, hi = int(a.min()), int(a.max())
            good = lo == hi == f.meta.frame_index % 256
            if use_views and not f.still_valid():
                stale += 1  # caught by the check after use, as a view's user must do
            elif good:
                ok += 1
            else:
                torn += 1
            del a, f
    out.put((ok, stale, torn))


@pytest.mark.parametrize("use_views", [False, True])
def test_no_torn_frame_is_ever_reported_valid(name, use_views):
    """A writer in one process, two readers in others, a tiny ring so the writer laps the
    readers constantly. Every frame a reader accepts must be whole."""
    ctx = mp.get_context("spawn")
    ready, out, count = ctx.Event(), ctx.Queue(), ctx.Value("q", 0)
    w = ctx.Process(target=_writer, args=(name, 2.5, ready, count))
    w.start()
    assert ready.wait(30)
    readers = [ctx.Process(target=_reader, args=(name, 1.5, use_views, out)) for _ in range(2)]
    for p in readers:
        p.start()
    results = [out.get(timeout=60) for _ in readers]
    for p in [*readers, w]:
        p.join(60)
    ok = sum(r[0] for r in results)
    torn = sum(r[2] for r in results)
    stale = sum(r[1] for r in results)
    print(f"\n  views={use_views}: {count.value} written, {ok} good, {stale} stale, {torn} torn", file=sys.stderr)
    assert torn == 0 and ok > 0


def test_rings_never_touch_the_resource_tracker(name, monkeypatch):
    # The tracker's shared bookkeeping caused KeyErrors and "leaked shared_memory" warnings
    # when writers and readers mixed in and across processes; rings manage their own lifetime.
    from multiprocessing import resource_tracker

    calls = []
    monkeypatch.setattr(resource_tracker, "register", lambda *a, **k: calls.append(("register", a)))
    monkeypatch.setattr(resource_tracker, "unregister", lambda *a, **k: calls.append(("unregister", a)))
    with RingWriter(name, slots=2, width=8, height=8) as w:
        w.write(np.zeros((8, 8, 3), np.uint8))
        with RingReader(name) as r:
            assert r.latest() is not None
    assert calls == []


def _write_then(name, how):
    w = RingWriter(name, slots=2, width=8, height=8)
    w.write(np.ones((8, 8, 3), np.uint8))
    if how == "crash":
        os._exit(1)  # no cleanup at all, like a segfault or kill -9
    # "exit": the interpreter exits normally without close()


def _exists(name):
    from maia_ring import _shm

    try:
        _shm.attach(name).close()
        return True
    except FileNotFoundError:
        return False


def test_normal_exit_removes_the_ring_crash_leaves_it_and_rm_clears_it(name):
    from maia_ring.cli import main

    ctx = mp.get_context("spawn")
    for how, survives in [("exit", False), ("crash", True)]:
        p = ctx.Process(target=_write_then, args=(name, how))
        p.start()
        p.join(30)
        assert _exists(name) is survives, how
    with RingWriter(name, slots=2, width=8, height=8) as w:  # a new writer replaces the stale ring
        with RingReader(name) as r:
            assert r.instance == w.instance
    p = ctx.Process(target=_write_then, args=(name, "crash"))
    p.start()
    p.join(30)
    main(["rm", name])
    assert not _exists(name)


def _segment(name):
    return f"/dev/shm/{name}"


@pytest.mark.skipif(not os.path.isdir("/dev/shm"), reason="Linux /dev/shm only")
def test_rings_are_private_unless_a_group_is_given(name, monkeypatch):
    monkeypatch.delenv("MAIA_RING_GID", raising=False)
    with RingWriter(name, slots=2, width=4, height=4):
        assert os.stat(_segment(name)).st_mode & 0o777 == 0o600
    gid = os.getgid()
    with RingWriter(name, slots=2, width=4, height=4, group=gid):
        st = os.stat(_segment(name))
        assert st.st_mode & 0o777 == 0o660 and st.st_gid == gid
    monkeypatch.setenv("MAIA_RING_GID", str(gid))
    with RingWriter(name, slots=2, width=4, height=4):
        assert os.stat(_segment(name)).st_mode & 0o777 == 0o660


def _read_as(uid, gid, ring, q):
    os.setgroups([gid])
    os.setgid(gid)
    os.setuid(uid)
    try:
        with RingReader(ring) as r:
            q.put(("ok", int(r.latest().array[0, 0, 0])))
    except PermissionError as exc:
        q.put(("denied", str(exc)))


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() != 0, reason="needs root to switch users")
def test_a_writer_running_as_root_shares_with_a_host_user_by_group(name, monkeypatch):
    # The Docker case: the gateway writes as root, a script on the host reads as a normal user.
    monkeypatch.delenv("MAIA_RING_GID", raising=False)
    user, group = 65534, 65534  # nobody / nogroup
    ctx = mp.get_context("fork")
    for share, expect in ((None, "denied"), (group, "ok")):
        with RingWriter(name, slots=2, width=4, height=4, group=share) as w:
            w.write(np.full((4, 4, 3), 7, np.uint8))
            q = ctx.Queue()
            p = ctx.Process(target=_read_as, args=(user, group, name, q))
            p.start()
            p.join(10)
            status, detail = q.get(timeout=5)
            assert status == expect, detail
            if status == "denied":
                assert "MAIA_RING_GID" in detail
            else:
                assert detail == 7
