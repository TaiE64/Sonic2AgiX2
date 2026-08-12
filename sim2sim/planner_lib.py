"""SONIC locomotion planner — Python driver (pure logic, no rendering).

Wraps `planner_sonic.onnx` and ports the parts of localmotion_kplanner*.hpp that
the deploy's C++ base class does around it: standing-context seeding, 30->50 Hz
resampling, and the replan/context-stitch used to grow a long trajectory.  Kept
free of mujoco/cv2 so both the G1 viz (planner_rollout.py) and the closed-loop
X2 integration (planner_drive_x2.py) can import it.

ONNX contract (introspected; matches the C++ headers):
  inputs : context_mujoco_qpos [1,4,36] (4 ctx frames x [3 pos|4 quat wxyz|29 joints], G1-MJ)
           target_vel[1] mode[1] movement_direction[1,3] facing_direction[1,3]
           random_seed[1] has_specific_target[1,1] specific_target_positions[1,4,3]
           specific_target_headings[1,4] allowed_pred_num_tokens[1,11] height[1]
  outputs: mujoco_qpos [1,64,36] (world frame, G1-MJ) | num_pred_frames [1] (int32)
  Native output rate = 30 Hz.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

# Portable defaults. Both ONNX paths can still be overridden from the CLI.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_DEPLOY_PLANNER = (
    _REPO
    / "GR00T-WholeBodyControl/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx"
)
_BUNDLED_PLANNER = _HERE / "models/planner_sonic.onnx"
PLANNER = os.environ.get(
    "SONIC_PLANNER_ONNX",
    str(_BUNDLED_PLANNER if _BUNDLED_PLANNER.exists() else _DEPLOY_PLANNER),
)
G1_XML = os.environ.get(
    "G1_MJCF",
    str(
        _REPO
        / "GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml"
    ),
)
X2_XML = os.environ.get("X2_MJCF", str(_REPO / "GMR" / "assets" / "agibot_x2" / "x2_mocap.xml"))

DEFAULT_HEIGHT = 0.788740          # PlannerConfig.default_height
NATIVE_FPS = 30.0
MODE_NAMES = {0: "IDLE", 1: "SLOW_WALK", 2: "WALK", 3: "RUN"}


class Planner:
    def __init__(self, onnx_path=PLANNER):
        self.s = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.inames = {i.name for i in self.s.get_inputs()}

    def run(self, context, mode, movement, facing, speed=-1.0, height=-1.0, seed=1234):
        """context: (4,36). Returns (T,36) world-frame G1-MJ qpos (T = valid frames)."""
        feed = {
            "context_mujoco_qpos": context[None].astype(np.float32),
            "target_vel": np.array([speed], np.float32),
            "mode": np.array([mode], np.int64),
            "movement_direction": np.array([movement], np.float32),
            "facing_direction": np.array([facing], np.float32),
            "random_seed": np.array([seed], np.int64),
            "has_specific_target": np.zeros((1, 1), np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), np.float32),
            "specific_target_headings": np.zeros((1, 4), np.float32),
            "allowed_pred_num_tokens": np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], np.int64),
            "height": np.array([height], np.float32),
        }
        feed = {k: v for k, v in feed.items() if k in self.inames}
        qpos, nframes = self.s.run(["mujoco_qpos", "num_pred_frames"], feed)
        n = int(nframes.reshape(-1)[0])
        return qpos[0, :n].astype(np.float64)


def standing_context(joints29):
    """4 identical standing frames (InitializeContext): xy=0, z=h, identity quat."""
    frame = np.concatenate([[0.0, 0.0, DEFAULT_HEIGHT, 1.0, 0.0, 0.0, 0.0], joints29])
    return np.tile(frame, (4, 1))


# --- quaternion / resampling (wxyz) -------------------------------------------
def quat_slerp(q0, q1, t):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    d = np.dot(q0, q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d) * t
    q2 = q1 - q0 * d
    q2 = q2 / np.linalg.norm(q2)
    return q0 * np.cos(th) + q2 * np.sin(th)


def resample_50hz(qpos30):
    """Port of ResampleGeneratedSequence50Hz (kept in G1-MJ order, no IL reorder)."""
    T = len(qpos30)
    n50 = int(np.floor((T / NATIVE_FPS) * 50))
    out = np.zeros((n50, 36))
    for f in range(n50):
        f30 = (f / 50.0) * NATIVE_FPS
        f0 = int(np.floor(f30))
        f1 = min(f0 + 1, T - 1)
        w1 = f30 - f0
        out[f, :3] = (1 - w1) * qpos30[f0, :3] + w1 * qpos30[f1, :3]
        out[f, 3:7] = quat_slerp(qpos30[f0, 3:7], qpos30[f1, 3:7], w1)
        out[f, 7:] = (1 - w1) * qpos30[f0, 7:] + w1 * qpos30[f1, 7:]
    return out


def context_from_motion(motion50, gen_frame):
    """Port of UpdateContextFromMotion: 4 ctx frames @30Hz sampled from the 50Hz motion."""
    ctx = np.zeros((4, 36))
    T = len(motion50)
    for n in range(4):
        f50 = (gen_frame / 50.0 + n / 30.0) * 50.0
        f0 = min(int(np.floor(f50)), T - 1)
        f1 = min(f0 + 1, T - 1)
        w1 = f50 - f0
        ctx[n, :3] = (1 - w1) * motion50[f0, :3] + w1 * motion50[f1, :3]
        ctx[n, 3:7] = quat_slerp(motion50[f0, 3:7], motion50[f1, 3:7], w1)
        ctx[n, 7:] = (1 - w1) * motion50[f0, 7:] + w1 * motion50[f1, 7:]
    return ctx


# --- rollout ------------------------------------------------------------------
FORWARD = (1.0, 0.0, 0.0)          # movement/facing in the planner's zero-yaw frame


def rollout_once(pl, mode, speed=-1.0, height=-1.0, seed=1234):
    """Single planner call from a straight-leg stand. Returns (T,36) G1-MJ @30Hz."""
    mv = np.asarray(FORWARD)
    q30 = pl.run(standing_context(np.zeros(29)), mode, mv, mv, speed, height, seed)
    if len(q30) == 0 or np.isnan(q30).any():
        raise RuntimeError("planner returned no frames / NaNs")
    return q30


def chained_rollout(pl, mode, seconds, speed=-1.0, height=-1.0, seed=1234):
    """Grow one world-frame 50Hz timeline of G1-MJ qpos by chained replans.

    Each replan seeds 4 context frames (30Hz spacing) ending at the timeline tip,
    runs the planner, and appends the new segment re-anchored (xy translation) to
    the tip so the splice is seamless.  A turning gait would also need a yaw
    re-anchor — the same bookkeeping maob needs in the closed-loop integration.
    Returns (T,36) G1-MJ @50Hz.
    """
    mv = np.asarray(FORWARD)
    timeline = resample_50hz(rollout_once(pl, mode, speed, height, seed))
    total = int(seconds * 50)
    while len(timeline) < total:
        ctx = context_from_motion(timeline, len(timeline) - 6)
        q30 = pl.run(ctx, mode, mv, mv, speed, height, seed)
        if len(q30) < 2 or np.isnan(q30).any():
            break
        seg = resample_50hz(q30)
        seg[:, :2] += timeline[-1, :2] - seg[0, :2]      # re-anchor xy to the tip
        timeline = np.concatenate([timeline, seg[1:]], axis=0)
    return timeline[:total]
