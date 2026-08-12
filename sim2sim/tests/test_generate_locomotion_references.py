from __future__ import annotations

import pathlib
import sys

import numpy as np


SIM2SIM_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM2SIM_ROOT))

from generate_locomotion_references import (  # noqa: E402
    DEFAULT_SPEEDS,
    build_reference,
    default_yaw_rates,
    integrate_body_twist,
    max_joint_speed,
)


def test_constant_body_twist_integrates_to_expected_circle():
    fps = 50.0
    xy, yaw = integrate_body_twist(101, fps, vx=0.4, vy=0.0, yaw_rate=0.5)
    assert np.isclose(yaw[-1], 1.0)
    assert np.allclose(xy[-1], [0.8 * np.sin(1.0), 0.8 * (1.0 - np.cos(1.0))])


def test_reference_exposes_only_29_controlled_joints():
    frames = 20
    qpos = np.zeros((frames, 36), dtype=np.float64)
    qpos[:, 2] = 0.79
    qpos[:, 3] = 1.0
    reference, _ = build_reference(qpos, vx=0.4, vy=0.0, yaw_rate=0.3, seed=7)
    assert reference["dof_pos"].shape == (frames, 29)
    assert len(reference["joint_names"]) == 29
    assert not any("head_" in name for name in reference["joint_names"])
    assert np.allclose(reference["command"], [0.4, 0.0, 0.3])


def test_reference_root_is_ground_snapped_to_x2_geometry():
    frames = 20
    qpos = np.zeros((frames, 36), dtype=np.float64)
    qpos[:, 2] = 0.79  # planner G1-height root: feet float on the X2
    qpos[:, 3] = 1.0
    reference, qc = build_reference(qpos, vx=0.4, vy=0.0, yaw_rate=0.0, seed=7)
    assert abs(qc["post_snap_sole_min_m"]) < 1e-9
    assert qc["ground_contact_fraction"] == 1.0
    assert np.allclose(
        reference["root_pos"][:, 2], 0.79 - qc["ground_snap_m"], atol=1e-6
    )


def test_max_joint_speed_is_measured_after_x2_mapping():
    reference = {
        "fps": 50.0,
        "dof_pos": np.array([[0.0, 0.0], [0.2, -0.1]], dtype=np.float32),
    }
    assert np.isclose(max_joint_speed(reference), 10.0)


def test_default_grid_covers_planner_run_range_with_safer_fast_turns():
    assert DEFAULT_SPEEDS[-1] == 2.0
    assert default_yaw_rates(1.0) == (-0.6, -0.3, 0.0, 0.3, 0.6)
    assert default_yaw_rates(1.2) == (-0.45, -0.3, 0.0, 0.3, 0.45)
    assert default_yaw_rates(2.0) == (-0.3, 0.0, 0.3)
