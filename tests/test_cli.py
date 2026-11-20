import json

import numpy as np

from maia_ring import RingWriter
from maia_ring.cli import main


def test_bench_runs(capsys):
    main(["bench", "--sizes", "720p", "-n", "3", "--json"])
    row = json.loads(capsys.readouterr().out)["rows"][0]
    assert row["size"] == "720p" and row["read_view_us"] < row["read_copy_us"]


def test_info_on_a_live_ring(capsys):
    with RingWriter("clitest", slots=4, width=64, height=48) as w:
        w.write(np.zeros((48, 64, 3), np.uint8))
        main(["info", "clitest"])
    info = json.loads(capsys.readouterr().out)
    assert info["slots"] == 4 and info["frames_written"] == 1 and info["size"] == "64x48"
