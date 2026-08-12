"""Generate command-exact, 29-DoF X2 references from the SONIC planner.

The planner supplies the gait joint trajectory.  Root XY and yaw are generated
by analytically integrating the requested body-frame twist, so reference
``vx/vy/wz`` cannot silently disagree with the training command.  The output
contains only the 29 policy-controlled X2 joints; head joints are not part of
this reference interface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ANY2ANY_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ANY2ANY_ROOT))

import planner_lib as PL  # noqa: E402
from kinematic_alignment import (  # noqa: E402
    X2_MUJOCO_JOINTS,
    g1_to_x2,
)
from retarget.build_locomotion_reference import X2FootKinematics  # noqa: E402


CONTROLLED_JOINT_NAMES = tuple(X2_MUJOCO_JOINTS[:29])
FPS = 50.0
PLANNER_CONTEXT_FRAMES = 4
DEFAULT_SPEEDS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0)


def integrate_body_twist(
    frames: int,
    fps: float,
    vx: float,
    vy: float,
    yaw_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact SE(2) integration of a constant body-frame twist."""
    time = np.arange(frames, dtype=np.float64) / fps
    yaw = yaw_rate * time
    xy = np.empty((frames, 2), dtype=np.float64)
    if abs(yaw_rate) < 1e-8:
        xy[:, 0] = vx * time
        xy[:, 1] = vy * time
    else:
        sine = np.sin(yaw)
        cosine = np.cos(yaw)
        xy[:, 0] = (vx * sine + vy * (cosine - 1.0)) / yaw_rate
        xy[:, 1] = (vx * (1.0 - cosine) + vy * sine) / yaw_rate
    return xy, yaw


def replace_root_planar_motion(
    planner_qpos: np.ndarray,
    vx: float,
    vy: float,
    yaw_rate: float,
    fps: float = FPS,
) -> np.ndarray:
    """Keep planner height/tilt while imposing command-exact XY and yaw."""
    qpos = np.asarray(planner_qpos, dtype=np.float64).copy()
    xy, yaw = integrate_body_twist(len(qpos), fps, vx, vy, yaw_rate)
    qpos[:, :2] = xy

    planner_xyzw = qpos[:, [4, 5, 6, 3]]
    euler = Rotation.from_quat(planner_xyzw).as_euler("xyz")
    euler[:, 2] = yaw
    xyzw = Rotation.from_euler("xyz", euler).as_quat()
    qpos[:, 3:7] = xyzw[:, [3, 0, 1, 2]]
    for frame in range(1, len(qpos)):
        if np.dot(qpos[frame - 1, 3:7], qpos[frame, 3:7]) < 0.0:
            qpos[frame, 3:7] *= -1.0
    return qpos


def planner_rollout(
    planner: PL.Planner,
    vx: float,
    vy: float,
    yaw_rate: float,
    seconds: float,
    mode: int,
    seed: int,
) -> np.ndarray:
    """Chain planner calls while steering toward the requested body twist."""
    target_frames = int(round(seconds * FPS))
    context = PL.standing_context(np.zeros(29, dtype=np.float64))
    timeline = np.empty((0, 36), dtype=np.float64)
    speed = float(np.hypot(vx, vy))
    body_direction = float(np.arctan2(vy, vx)) if speed > 1e-8 else 0.0
    replan = 0
    while len(timeline) < target_frames:
        current_yaw = yaw_rate * len(timeline) / FPS
        movement_yaw = current_yaw + body_direction
        facing_yaw = current_yaw + float(np.clip(yaw_rate, -1.0, 1.0))
        movement = np.array(
            [np.cos(movement_yaw), np.sin(movement_yaw), 0.0], dtype=np.float64
        )
        facing = np.array(
            [np.cos(facing_yaw), np.sin(facing_yaw), 0.0], dtype=np.float64
        )
        qpos30 = planner.run(
            context,
            mode,
            movement,
            facing,
            speed=speed,
            # Keep the stochastic gait identity/phase stable across replans.
            # Changing the seed here causes discontinuous RUN knee phases.
            seed=seed,
        )
        if len(qpos30) < 2 or not np.isfinite(qpos30).all():
            raise RuntimeError(f"planner failed at replan {replan}")
        if len(timeline) == 0:
            segment = PL.resample_50hz(qpos30)
        else:
            # The first four native-rate frames reproduce the four input
            # context frames.  Only frames after that are a new prediction.
            # Anchor the continuation at the exact current tip because the
            # network's reconstruction of the last context frame is lossy.
            continuation30 = np.concatenate(
                [timeline[-1:], qpos30[PLANNER_CONTEXT_FRAMES:]], axis=0
            )
            segment = PL.resample_50hz(continuation30)
        if len(segment) < 2:
            raise RuntimeError(f"planner returned a short segment at replan {replan}")
        timeline = np.concatenate(
            [timeline, segment if len(timeline) == 0 else segment[1:]], axis=0
        )
        timeline = replace_root_planar_motion(timeline, vx, vy, yaw_rate)
        context = PL.context_from_motion(timeline, max(0, len(timeline) - 6))
        replan += 1
    return timeline[:target_frames]


_FOOT_KINEMATICS = None


def _foot_kinematics() -> X2FootKinematics:
    global _FOOT_KINEMATICS
    if _FOOT_KINEMATICS is None:
        _FOOT_KINEMATICS = X2FootKinematics()
    return _FOOT_KINEMATICS


def ground_snap(
    root_pos: np.ndarray,
    root_rot_wxyz: np.ndarray,
    dof_pos_29: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Drop the planner's G1-height root so X2 stance soles sit on the floor.

    The planner keeps the G1 root height (~0.78 m); the X2 straight-leg root
    height is only 0.73 m, so passing the height through leaves the feet
    floating in the air for the whole reference.
    """
    feet = _foot_kinematics().trajectory(root_pos, root_rot_wxyz, dof_pos_29)
    sole = feet[:, :, 2].min(axis=1)
    offset = float(np.percentile(sole, 5))
    snapped = np.asarray(root_pos, dtype=np.float64).copy()
    snapped[:, 2] -= offset
    return snapped, sole - offset, offset


def build_reference(
    planner_qpos: np.ndarray,
    vx: float,
    vy: float,
    yaw_rate: float,
    seed: int,
) -> tuple[dict, dict]:
    qpos = replace_root_planar_motion(planner_qpos, vx, vy, yaw_rate)
    x2_dof_31 = g1_to_x2(qpos[:, 7:])
    if not np.allclose(x2_dof_31[:, 29:], 0.0):
        raise AssertionError("head joints must remain neutral")
    root_pos, sole, snap_offset = ground_snap(
        qpos[:, :3], qpos[:, 3:7], x2_dof_31[:, :29]
    )
    command = np.tile(
        np.asarray([vx, vy, yaw_rate], dtype=np.float32), (len(qpos), 1)
    )
    mode = default_mode(float(np.hypot(vx, vy)))
    reference = {
        "robot": "agibot_x2",
        "fps": FPS,
        "root_pos": root_pos.astype(np.float32),
        "root_rot": qpos[:, 3:7].astype(np.float32),
        "dof_pos": x2_dof_31[:, :29].astype(np.float32),
        "joint_names": CONTROLLED_JOINT_NAMES,
        "command": command,
        "command_names": ("vx_body", "vy_body", "wz"),
        "locomotion_mode": PL.MODE_NAMES[mode].lower(),
        "source": f"sonic_planner:mode={PL.MODE_NAMES[mode]}:seed={seed}",
    }
    qc = {
        "ground_snap_m": snap_offset,
        "post_snap_sole_min_m": float(sole.min()),
        "ground_contact_fraction": float((sole <= 0.02).mean()),
    }
    return reference, qc


def max_joint_speed(reference: dict) -> float:
    """Maximum finite-difference speed after the G1-to-X2 mapping."""
    joints = np.asarray(reference["dof_pos"], dtype=np.float64)
    if len(joints) < 2:
        return 0.0
    return float(np.abs(np.diff(joints, axis=0)).max() * float(reference["fps"]))


def default_mode(speed: float) -> int:
    if speed <= 0.25:
        return 1
    if speed <= 0.70:
        return 2
    return 3


def default_yaw_rates(speed: float) -> tuple[float, ...]:
    """Use the planner's broad turn range while keeping fast runs feasible."""
    if speed <= 1.0:
        return (-0.6, -0.3, 0.0, 0.3, 0.6)
    if speed <= 1.5:
        return (-0.45, -0.3, 0.0, 0.3, 0.45)
    return (-0.3, 0.0, 0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", default=PL.PLANNER)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--speeds", type=float, nargs="+", default=DEFAULT_SPEEDS
    )
    parser.add_argument(
        "--yaw-rates",
        type=float,
        nargs="+",
        help="override the default speed-dependent yaw-rate grid",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1001, 1002, 1003, 1004])
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--max-joint-speed", type=float, default=15.0)
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    specs = [
        (speed, yaw_rate, seed)
        for speed in args.speeds
        for yaw_rate in (
            args.yaw_rates
            if args.yaw_rates is not None
            else default_yaw_rates(speed)
        )
        for seed in args.seeds
    ]
    if args.max_clips is not None:
        specs = specs[: args.max_clips]
    args.out_root.mkdir(parents=True, exist_ok=True)
    planner = PL.Planner(args.planner)
    report_path = args.out_root / "manifest.jsonl"
    passed = failed = 0
    report_mode = "w" if args.overwrite else "a"
    with report_path.open(report_mode, encoding="utf-8") as report_stream:
        for speed, yaw_rate, seed in specs:
            mode = default_mode(speed)
            name = f"vx{speed:+.2f}_wz{yaw_rate:+.2f}_seed{seed}.pkl"
            target = args.out_root / name
            if target.exists() and not args.overwrite:
                continue
            result = {
                "path": name,
                "command": [speed, 0.0, yaw_rate],
                "mode": PL.MODE_NAMES[mode].lower(),
                "seed": seed,
            }
            try:
                qpos = planner_rollout(
                    planner,
                    speed,
                    0.0,
                    yaw_rate,
                    args.seconds,
                    mode,
                    seed,
                )
                reference, qc = build_reference(qpos, speed, 0.0, yaw_rate, seed)
                result.update(qc)
                peak_speed = max_joint_speed(reference)
                if peak_speed > args.max_joint_speed:
                    raise RuntimeError(
                        f"X2 joint-speed spike {peak_speed:.3f} rad/s exceeds "
                        f"{args.max_joint_speed:.3f} rad/s"
                    )
                if qc["post_snap_sole_min_m"] < -0.06:
                    raise RuntimeError(
                        f"stance sole dips {qc['post_snap_sole_min_m']:.3f} m "
                        "below ground after snap"
                    )
                if qc["ground_contact_fraction"] < 0.15:
                    raise RuntimeError(
                        "reference has almost no ground contact "
                        f"({qc['ground_contact_fraction']:.2f})"
                    )
                with target.open("wb") as stream:
                    pickle.dump(reference, stream)
                result.update(
                    frames=len(qpos),
                    max_joint_speed_rad_s=peak_speed,
                    fail=None,
                )
                passed += 1
            except Exception as error:  # noqa: BLE001
                result.update(fail=f"{type(error).__name__}: {error}")
                failed += 1
            report_stream.write(json.dumps(result, sort_keys=True) + "\n")
            report_stream.flush()
            print(json.dumps(result, sort_keys=True))
    print(f"planner references: {passed} passed, {failed} failed -> {args.out_root}")


if __name__ == "__main__":
    main()
