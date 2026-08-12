from __future__ import annotations

import pathlib
import sys


TRAIN_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN_ROOT))

from build_mixed_motion_lib import locomotion_repeat_count  # noqa: E402


def test_repeat_count_reaches_requested_share_without_undershooting():
    repeats = locomotion_repeat_count(14069, 60, 0.30)
    share = repeats * 60 / (14069 + repeats * 60)
    previous_share = (repeats - 1) * 60 / (14069 + (repeats - 1) * 60)
    assert repeats == 101
    assert share >= 0.30
    assert previous_share < 0.30
