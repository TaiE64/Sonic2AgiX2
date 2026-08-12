"""Drive the any2any X2 tracking policy with the SONIC locomotion planner.

The reference (cmf/maob) that the policy tracks comes from the planner
(planner_sonic.onnx) instead of a pre-recorded clip.  Planner output is G1, fed
DIRECTLY as G1-IL reference (M@MT=I, so embedding to X2 and back is exact anyway).
Everything else — Align (Phi), ObsBuilder, Policy, MuJoCo physics — is reused
verbatim from the validated mujoco_sim2sim_x2 harness; only the reference SOURCE
is swapped.

Validation ladder (--ref):
  static  : constant standing G1 reference -> robot must stand still.  Isolates
            the obs contract / block order / action path from planner dynamics:
            if this falls, it is the wiring, not the planner.
  preroll : planner rolled out open-loop into a WALK/RUN reference -> robot should
            track it and walk.  Planner does not see the robot yet.
  online  : (stage 2, TODO) planner re-plans from the live robot state each
            replan period — the real deploy feedback loop.

Usage (byte isaaclab env):
  PY=/home/byte/miniconda3/envs/isaaclab/bin/python
  V21=.../logs_rl/TRL_X2_Track/v21/exported/model_step_012000_g1.onnx
  $PY planner_drive_x2.py --ref static  --onnx $V21 --block_order "cmf|maob|pro" \
      --steps 300 --render out/x2_static.mp4
  $PY planner_drive_x2.py --ref preroll --mode 2 --onnx $V21 --block_order "cmf|maob|pro" \
      --steps 400 --render out/x2_planner_walk.mp4 --ref_ghost
"""

from __future__ import annotations

import argparse
import os
import sys
import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_A2A = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _A2A)
sys.path.insert(0, os.path.join(_A2A, "any2any_train"))

import mujoco_sim2sim_x2 as S          # noqa: E402  (Align/ObsBuilder/Policy/build_model/...)
import planner_lib as PL               # noqa: E402  (Planner + rollout + paths)
from kinematic_alignment import G1_MUJOCO_TO_ISAACLAB_DOF  # noqa: E402

S.X2_XML = PL.X2_XML                    # repo scripts hardcode /home/taie; override for build_model
CAPTURE = os.path.join(_HERE, "capture")
G1_MJ2IL = np.asarray(G1_MUJOCO_TO_ISAACLAB_DOF)     # ref_il[:, j] = g1mj[:, G1_MJ2IL[j]]


def g1mj_to_il(g1mj):
    return g1mj[:, G1_MJ2IL]


def load_rel_bias(al, def_il):
    """rel_bias is kinematic (obs-construction offset), not policy-specific, so the
    legacy capture is valid across checkpoints.  Mirrors validate() section B."""
    p = os.path.join(CAPTURE, "rel_bias.npy")
    if os.path.exists(p):
        return np.load(p)
    gt = os.path.join(CAPTURE, "gt_step0.npz")
    if not os.path.exists(gt):
        return None
    s0 = np.load(gt, allow_pickle=True)
    jp0 = al.lin(s0["joint_pos_il"].astype(np.float64) - def_il)
    rb = s0["obs.actor_obs"][30:320].reshape(S.HIST, 29)[-1] - jp0
    np.save(p, rb)
    print(f"  measured rel_bias from legacy capture (|max|={np.abs(rb).max():.5f})")
    return rb


def preroll_planner(args):
    pl = PL.Planner(args.planner)
    traj = PL.chained_rollout(pl, args.mode, seconds=(args.steps + 60) / 50.0,
                              speed=args.speed, height=args.height, seed=args.seed)
    print(f"  planner preroll: {len(traj)} frames @50Hz ({len(traj)/50:.1f}s)")
    return traj


def build_reference(args, al, def_il):
    if args.ref == "static":
        T = args.steps + 60
        pos = np.tile(al.pos(def_il[None])[0], (T, 1))         # robot's own standing G1 pose
        rot = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (T, 1))
        trans = np.tile(np.array([0.0, 0.0, PL.DEFAULT_HEIGHT]), (T, 1))
    else:  # preroll
        traj = preroll_planner(args)
        pos = g1mj_to_il(traj[:, 7:])
        rot = traj[:, 3:7]
        trans = traj[:, :3]
    vel = np.diff(pos, axis=0, append=pos[-1:]) * S.TARGET_FPS
    x2il = al.OFF + pos @ al.M                                  # G1-IL -> X2-IL (M@MT=I)
    x2dof = x2il[:, S.IL2MJ]                                    # X2-MJ (reset + ghost)
    return dict(pos=pos, vel=vel, rot=rot, trans=trans, x2dof=x2dof, T=len(pos))


def run(args):
    import mujoco

    al = S.Align()
    C = np.load(os.path.join(CAPTURE, "constants.npz"), allow_pickle=True)
    def_il = C["default_joint_pos"].astype(np.float64)
    scale_il = C["actterm_joint_pos_scale"].astype(np.float64)
    off_il = C["actterm_joint_pos_offset"].astype(np.float64)
    rel_bias = load_rel_bias(al, def_il)

    model, xml_path, (lo, hi) = S.build_model(C)
    data = mujoco.MjData(model)
    pol = S.Policy(args.onnx)
    order = tuple(args.block_order.split("|"))
    print(f"  block order: [{' | '.join(order)}]   onnx: {os.path.basename(args.onnx)}")

    REF = build_reference(args, al, def_il)
    # reuse the validated ObsBuilder; feed it a stub clip, then override the
    # reference arrays with the planner's (cmf/maob read these + self.clip.rot).
    ob = S.ObsBuilder(al, types.SimpleNamespace(dof=np.zeros((2, 31)), T=2,
                      rot=np.array([[1.0, 0, 0, 0]] * 2)), def_il,
                      rel_bias=rel_bias, order=order)
    ob.ref_pos_g1 = REF["pos"]
    ob.ref_vel_g1 = REF["vel"]
    ob.clip = types.SimpleNamespace(T=REF["T"], rot=REF["rot"], trans=REF["trans"], dof=REF["x2dof"])

    # reset robot onto reference frame 0
    data.qpos[:3] = REF["trans"][0]
    data.qpos[3:7] = REF["rot"][0]
    data.qpos[7:] = REF["x2dof"][0]
    mujoco.mj_forward(model, data)

    nsub = int(round(S.CTRL_DT / S.PHYS_DT))
    render = bool(args.render)
    if render:
        os.environ.setdefault("MUJOCO_GL", "egl")
        import cv2
        w, h = (1440, 540) if args.ref_ghost else (720, 540)
        vw = cv2.VideoWriter(args.render, cv2.VideoWriter_fourcc(*"mp4v"), 50, (w, h))
        rend = mujoco.Renderer(model, 540, 720)
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation, cam.azimuth = args.cam_dist, -12, 140
        ref_data = mujoco.MjData(model) if args.ref_ghost else None

    prev_a = np.zeros(29)
    fall = None
    jerr_sum = 0.0
    for t in range(args.steps):
        f = t
        obs = ob.build(f, data.qpos[7:][S.MJ2IL], data.qvel[6:][S.MJ2IL],
                       data.qpos[3:7].copy(), data.qvel[3:6].copy(), prev_a)
        a = pol(obs)
        q_des = off_il + al.act_back(a) * scale_il
        data.ctrl[:] = np.clip(q_des[S.IL2MJ], lo, hi)
        for _ in range(nsub):
            mujoco.mj_step(model, data)
        prev_a = a

        dh = abs(data.qpos[2] - REF["trans"][f][2])
        qr = S.quat_to_mat(data.qpos[3:7]).T @ S.quat_to_mat(REF["rot"][f])
        ori = np.arccos(np.clip((np.trace(qr) - 1) / 2, -1, 1))
        jerr_sum += np.abs(data.qpos[7:] - REF["x2dof"][f]).mean()
        if fall is None and (dh > 0.30 or ori > 0.9):
            fall = t
            print(f"  [fall] t={t/50:.2f}s  h_err={dh:.3f}  ori_err={np.degrees(ori):.1f}deg")
        if t % 100 == 0:
            print(f"  t={t/50:5.2f}s  h={data.qpos[2]:.3f}(ref {REF['trans'][f][2]:.3f})  "
                  f"ori={np.degrees(ori):5.1f}deg  root_xy={data.qpos[0]:+.2f},{data.qpos[1]:+.2f}")

        if render:
            if args.cam_fixed:
                cam.lookat[:] = [args.cam_fixed, 0.0, 0.7]
            else:
                cam.lookat[:] = data.qpos[:3]
            rend.update_scene(data, cam)
            frame = rend.render()
            if args.ref_ghost:
                ref_data.qpos[:3] = REF["trans"][f]
                ref_data.qpos[3:7] = REF["rot"][f]
                ref_data.qpos[7:] = REF["x2dof"][f]
                mujoco.mj_forward(model, ref_data)
                if not args.cam_fixed:
                    cam.lookat[:] = REF["trans"][f]
                rend.update_scene(ref_data, cam)
                frame = np.concatenate([frame, rend.render()], axis=1)
            vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if render:
        vw.release()
        print(f"  video: {args.render}")
    surv = (fall if fall is not None else args.steps) / 50.0
    print(f"RESULT ref={args.ref} survived={surv:.2f}s/{args.steps/50:.2f}s "
          f"({'FELL' if fall is not None else 'upright'}) "
          f"root_xy_final=({data.qpos[0]:+.2f},{data.qpos[1]:+.2f}) "
          f"mean_jerr={jerr_sum/args.steps:.3f}rad")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", choices=["static", "preroll"], default="static")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--block_order", default="cmf|maob|pro")
    ap.add_argument("--planner", default=PL.PLANNER)
    ap.add_argument("--mode", type=int, default=2)
    ap.add_argument("--speed", type=float, default=-1.0)
    ap.add_argument("--height", type=float, default=-1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--render", default="")
    ap.add_argument("--ref_ghost", action="store_true")
    ap.add_argument("--cam_dist", type=float, default=3.0)
    ap.add_argument("--cam_fixed", type=float, default=0.0,
                    help="fix camera lookat at (X,0,0.7) so forward travel is visible (0=follow)")
    args = ap.parse_args()
    print(f"ref={args.ref}  mode={args.mode}({PL.MODE_NAMES.get(args.mode,'?')})  steps={args.steps}")
    run(args)


if __name__ == "__main__":
    main()
