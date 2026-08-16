"""Bake the human walk cycle into plain G1-MJ arrays for on-robot use.

`planner_teleop_x2.py` loads `human_walk_cycle.npz` (X2-MJ joints) and converts
it to G1 through the kinematic alignment at startup -- which pulls in mujoco and
the training-side alignment code. The robot only has stock python3, so the
conversion is done once here and the result shipped as a plain array file that
`planner_refsrc_x2.py` can np.load with nothing but numpy.

Neither .npz is redistributable (AMASS terms), so both stay out of git; this
script is the reproducible path from the source clip to the baked file.

    python bake_walk_cycle.py                     # human_walk_cycle.npz -> human_walk_cycle_g1mj.npz
    python bake_walk_cycle.py --in X --out Y

Output arrays:
    cycle  (M,36) float64  50 Hz, one loop, G1-MJ [3 root pos | 4 quat wxyz | 29 joints]
    stride ()     float64  metres advanced per loop, so xy can integrate across loops
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco_sim2sim_x2 as S  # noqa: E402
import planner_lib as PL  # noqa: E402
from kinematic_alignment import G1_MUJOCO_TO_ISAACLAB_DOF as G1_MJ2IL  # noqa: E402


def bake(src_path, out_path):
    z = np.load(src_path)
    dof31 = z["dof"].astype(np.float64)                  # X2-MJ joints
    pos = z["root_pos"].astype(np.float64)
    rot = z["root_rot_wxyz"].astype(np.float64)
    fps, stride, n_src = float(z["fps"]), float(z["stride"]), len(dof31)

    al = S.Align()
    g1 = np.zeros((n_src, 29))
    for i in range(n_src):
        g1[i][G1_MJ2IL] = al.pos(dof31[i][S.MJ2IL][None])[0]

    # Close the loop: the frame after the last one is the first one, advanced by
    # a full stride, so the resample never interpolates across the seam wrongly.
    src = np.concatenate([pos, rot, g1], axis=1)
    wrap = src[0].copy()
    wrap[0] += stride
    src = np.vstack([src, wrap])

    n50 = int(round(n_src / fps * 50.0))
    cyc = np.zeros((n50, 36))
    for f in range(n50):
        x = f / n50 * n_src
        f0 = int(x)
        w = x - f0
        cyc[f, :3] = (1 - w) * src[f0, :3] + w * src[f0 + 1, :3]
        cyc[f, 3:7] = PL.quat_slerp(src[f0, 3:7], src[f0 + 1, 3:7], w)
        cyc[f, 7:] = (1 - w) * src[f0, 7:] + w * src[f0 + 1, 7:]

    np.savez(out_path, cycle=cyc, stride=stride, src_fps=fps, src_frames=n_src)
    speed = stride / (n50 / 50.0)
    print(f"{src_path} ({n_src} frames @{fps:g}Hz) -> {out_path}")
    print(f"  {n50} frames @50Hz, {n50 / 50:.3f}s loop, stride {stride:.4f} m -> {speed:.3f} m/s")
    return cyc, stride


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default=os.path.join(_HERE, "human_walk_cycle.npz"))
    ap.add_argument("--out", dest="out", default=os.path.join(_HERE, "human_walk_cycle_g1mj.npz"))
    args = ap.parse_args()
    bake(args.src, args.out)


if __name__ == "__main__":
    main()
