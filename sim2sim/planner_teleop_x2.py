"""Interactive teleop: drive X2 with the SONIC planner in real time.

Online closed loop (the real deploy shape): the planner holds a rolling PLAN;
every replan tick it continues that plan from a look-ahead point with your
current command and re-splices forward.  The V21 policy tracks the plan in
MuJoCo physics.  (Seeding the planner from its own plan — not the robot's live
pose — is what makes the gait progress instead of marching in place; the policy
handles the physical tracking / disturbances.)

Only the reference JOINTS (cmf) and root ORIENTATION (maob) drive the policy —
root translation is never consumed.  The plan is world-frame anchored at the
standing start, so its root orientation is directly comparable to the robot's
and the robot turns when you steer (facing command curves the plan).

Keys (own GLFW window; polled hold-state, no collision with viewer built-ins):
  hold W        walk forward       hold Shift+W   run
  hold S        walk backward      release all    stand (idle)
  hold A / D    turn left / right  (release = hold current heading)
  + / -         trim speed (planner target_vel)
  mouse drag    orbit camera       scroll zoom    Esc quit

Run interactive (on a machine with a display):
  PY=/home/byte/miniconda3/envs/isaaclab/bin/python
  V21=.../v21/exported/model_step_012000_g1.onnx
  $PY planner_teleop_x2.py --onnx $V21

Headless self-test (scripted commands -> mp4, no window):
  MUJOCO_GL=egl $PY planner_teleop_x2.py --onnx $V21 --script --render out/teleop_selftest.mp4
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "any2any_train"))

CONTEXT_FRAMES = 4  # native-rate frames the planner spends reconstructing its input context

# Forward walk uses SLOW_WALK pushed to 0.9 m/s: 1.8 steps/s and ~1.0 m
# strides, close to the human gait profile.  The WALK mode minces at 3.2
# steps/s regardless of speed (G1-native cadence, looks hurried on the X2).
SLOW_WALK_SPEED = 0.9

# Loopable human walk cycle (AMASS B3 straight walk retargeted to X2); when
# present, forward walk tracks this instead of the planner's slow_walk, whose
# gait is left/right asymmetric at the source (knee ROM 61/72 deg).
HUMAN_WALK_CYCLE = os.path.join(_HERE, "human_walk_cycle.npz")
CYCLE_SWITCH_BLEND = 10  # 50Hz frames to morph when entering the cycle


def planner_speed(cmd):
    return SLOW_WALK_SPEED if (cmd.mode == 1 and cmd.speed < 0) else cmd.speed

import mujoco_sim2sim_x2 as S          # noqa: E402
import planner_lib as PL               # noqa: E402
from kinematic_alignment import G1_MUJOCO_TO_ISAACLAB_DOF  # noqa: E402

S.X2_XML = PL.X2_XML
G1_MJ2IL = np.asarray(G1_MUJOCO_TO_ISAACLAB_DOF)      # il[j] = mj[G1_MJ2IL[j]]
CAPTURE = os.path.join(_HERE, "capture")
DEFAULT_POLICY = os.path.join(
    os.path.dirname(os.path.dirname(_HERE)),
    "GR00T-WholeBodyControl", "logs_rl", "X2_Locomotion",
    "exported", "model_step_004000_g1.onnx",
)

# --- quaternion helpers (wxyz) ------------------------------------------------
FORWARD = np.array([1.0, 0.0, 0.0])          # planner always walks straight; heading is synthesized


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rot_quats(q, yaw):
    """Rz(yaw) ⊗ q for an (N,4) wxyz array — rotate the maob heading target."""
    cw, sw = math.cos(yaw / 2), math.sin(yaw / 2)
    o = np.empty_like(q)
    o[:, 0] = cw * q[:, 0] - sw * q[:, 3]
    o[:, 1] = cw * q[:, 1] - sw * q[:, 2]
    o[:, 2] = cw * q[:, 2] + sw * q[:, 1]
    o[:, 3] = cw * q[:, 3] + sw * q[:, 0]
    return o


class Command:
    def __init__(self, speed=-1.0):
        self.mode = 0            # IDLE
        self.speed = speed       # target_vel (-1 = mode default)
        self.move = np.array([1.0, 0.0, 0.0])   # planner movement_direction (+x fwd / -x back)
        self.turn = 0.0          # commanded YAW RATE (rad/s); + = left.  The policy tracks
        self.dirty = True        # orientation, not angular velocity, so we integrate this
        #                          into a heading and rotate the maob target (see control_step).

    def label(self):
        return f"mode={self.mode}({PL.MODE_NAMES.get(self.mode,'?')}) speed={self.speed:+.1f} turn={math.degrees(self.turn):+.0f}deg/s"


class ReferenceManager:
    """Online planner reference.  Holds the current PLAN as a G1-MJ trajectory
    (world frame, @50Hz) and — crucially — re-seeds each replan from the plan's
    own continuation (context_from_motion), NOT from the robot's live state.
    Seeding from the robot makes the planner restart from stand every tick and
    march in place; seeding from the plan lets the gait progress (matches the
    deploy's UpdateContextFromMotion / chained_rollout).  Command changes take
    effect by truncating the plan near the playhead and re-planning forward.

    cmf reads self.pos (G1-IL joints); maob reads self.rot (world root quat)."""

    def __init__(self, planner, al):
        self.pl = planner
        self.al = al
        self.plan = None                     # (N,36) G1-MJ world @50Hz
        self.pos = self.vel = self.rot = None
        self.T = 0
        self.heading = 0.0                   # synthesized commanded yaw (rad); integrated from cmd.turn
        self.cycle = None                    # (M,36) G1-MJ human walk cycle @50Hz, +X heading
        self.cycle_stride = 0.0
        self._plan_phase = []                # per plan frame: cycle index, or -1 for planner frames
        self._load_cycle()

    def _load_cycle(self):
        if not os.path.exists(HUMAN_WALK_CYCLE):
            return
        z = np.load(HUMAN_WALK_CYCLE)
        dof31 = z["dof"].astype(np.float64)              # X2-MJ joints
        pos = z["root_pos"].astype(np.float64)
        rot = z["root_rot_wxyz"].astype(np.float64)
        fps, stride, n_src = float(z["fps"]), float(z["stride"]), len(dof31)
        g1 = np.zeros((n_src, 29))
        for i in range(n_src):
            g1_il = self.al.pos(dof31[i][S.MJ2IL][None])[0]
            g1[i][G1_MJ2IL] = g1_il
        src = np.concatenate([pos, rot, g1], axis=1)
        wrap = src[0].copy()
        wrap[0] += stride                                # cycle end == shifted cycle start
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
        self.cycle, self.cycle_stride = cyc, stride

    @staticmethod
    def _z_rotate(yaw, quats):
        """Compose R_z(yaw) with wxyz quaternion rows."""
        w, z = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
        qw, qx, qy, qz = np.asarray(quats).T
        return np.stack(
            [w * qw - z * qz, w * qx - z * qy, w * qy + z * qx, w * qz + z * qw],
            axis=1,
        )

    def _cycle_dx(self, a):
        """Cycle-local displacement from frame a to frame (a+1) % M."""
        m = len(self.cycle)
        b = (a + 1) % m
        d = self.cycle[b, :2] - self.cycle[a, :2]
        if b == 0:
            d = d + np.array([self.cycle_stride, 0.0])
        return d

    def _extend_cycle(self):
        """Append one human-walk cycle at the plan tip instead of a planner
        segment; joints and root orientation loop, xy integrates the cycle's
        own step displacement rotated into the tip heading."""
        gen = self.T - 1
        tip = self.plan[gen]
        last = self._plan_phase[gen]
        switching = last < 0
        m = len(self.cycle)
        start = 0 if switching else (last + 1) % m
        idx = [(start + k) % m for k in range(m)]

        qw, qx, qy, qz = tip[3:7]
        tip_yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        cyc_yaw = np.arctan2(
            2 * (self.cycle[start, 3] * self.cycle[start, 6]
                 + self.cycle[start, 4] * self.cycle[start, 5]),
            1 - 2 * (self.cycle[start, 5] ** 2 + self.cycle[start, 6] ** 2),
        )
        spin = tip_yaw - cyc_yaw
        rot_mat = np.array([[np.cos(spin), -np.sin(spin)], [np.sin(spin), np.cos(spin)]])

        seg = self.cycle[idx].copy()
        seg[:, 3:7] = self._z_rotate(spin, seg[:, 3:7])
        prev_idx = (start - 1) % m
        xy = tip[:2].copy()
        for k, cyc_i in enumerate(idx):
            a = prev_idx if k == 0 else idx[k - 1]
            xy = xy + rot_mat @ self._cycle_dx(a)
            seg[k, :2] = xy
        if switching:
            for j in range(min(CYCLE_SWITCH_BLEND, m)):
                alpha = (j + 1) / (CYCLE_SWITCH_BLEND + 1)
                seg[j, 7:] = (1 - alpha) * tip[7:] + alpha * seg[j, 7:]
                seg[j, 2] = (1 - alpha) * tip[2] + alpha * seg[j, 2]
                q0, q1 = tip[3:7], seg[j, 3:7]
                if np.dot(q0, q1) < 0:
                    q1 = -q1
                q = (1 - alpha) * q0 + alpha * q1
                seg[j, 3:7] = q / np.linalg.norm(q)
        self.plan = np.concatenate([self.plan, seg], axis=0)
        self._plan_phase.extend(idx)
        self._derive()
        return True

    def _robot_context(self, robot_qpos):
        """4 context frames from the robot's live pose (Initialize-style, world quat)."""
        g1_il = self.al.pos(robot_qpos[7:][S.MJ2IL][None])[0]
        g1_mj = np.zeros(29)
        g1_mj[G1_MJ2IL] = g1_il
        frame = np.concatenate([robot_qpos[:3], robot_qpos[3:7], g1_mj])
        return np.tile(frame, (4, 1))

    def _derive(self):
        self.pos = self.plan[:, 7:][:, G1_MJ2IL]            # G1-IL joints
        self.vel = np.diff(self.pos, axis=0, append=self.pos[-1:]) * 50.0
        self.rot = self.plan[:, 3:7]                        # world root quat
        self.T = len(self.plan)

    def init(self, robot_qpos, cmd):
        q30 = self.pl.run(self._robot_context(robot_qpos), cmd.mode, cmd.move,
                          FORWARD, planner_speed(cmd), -1.0, 1234)
        self.plan = PL.resample_50hz(q30)
        self._plan_phase = [-1] * len(self.plan)
        self._derive()

    def _extend(self, cmd):
        """Append one planner segment at the plan tip (chained_rollout style).
        Seeding from the tip — a steady-state gait pose — is what keeps the robot
        progressing instead of replaying the start-from-stand transient."""
        if cmd.mode == 1 and self.cycle is not None:
            return self._extend_cycle()
        gen = self.T - 1
        ctx = PL.context_from_motion(self.plan, max(0, gen - 5))
        q30 = self.pl.run(ctx, cmd.mode, cmd.move, FORWARD, planner_speed(cmd), -1.0, 1234)
        if len(q30) < CONTEXT_FRAMES + 2 or np.isnan(q30).any():
            return False
        # The first CONTEXT_FRAMES native frames are the planner's lossy
        # reconstruction of its input context; splicing them in steps every
        # joint at the seam and the policy hitches when the step enters its
        # future window.  Anchor the continuation at the exact plan tip and
        # drop the reconstruction instead (same treatment as
        # generate_locomotion_references.planner_rollout).
        continuation30 = np.concatenate(
            [self.plan[gen:gen + 1], q30[CONTEXT_FRAMES:]], axis=0
        )
        seg = PL.resample_50hz(continuation30)
        seg[:, :2] += self.plan[gen, :2] - seg[0, :2]        # xy re-anchor (heading via context)
        self.plan = np.concatenate([self.plan, seg[1:]], axis=0)
        self._plan_phase.extend([-1] * (len(seg) - 1))
        self._derive()
        return True

    def update(self, t, cmd, changed):
        """Keep the plan filled ~1s ahead of the playhead.  On a command change,
        discard the stale far-future plan and re-plan from just ahead of the
        playhead so the new command takes effect in ~0.2s (not ~1s)."""
        if changed and self.T > t + 12:
            self.plan = self.plan[:t + 12]
            self._plan_phase = self._plan_phase[:t + 12]
            self._derive()
        guard = 0
        while self.T < t + 60 and guard < 5:
            if not self._extend(cmd):
                break
            guard += 1


def setup(args):
    import mujoco
    al = S.Align()
    C = S.load_runtime_constants(CAPTURE)
    def_il = C["default_joint_pos"].astype(np.float64)
    scale_il = C["actterm_joint_pos_scale"].astype(np.float64)
    off_il = C["actterm_joint_pos_offset"].astype(np.float64)
    rel_bias = C.get("rel_bias")
    model, _, (lo, hi) = S.build_model(C)
    data = mujoco.MjData(model)
    pol = S.Policy(args.onnx)
    if args.block_order:
        order = tuple(args.block_order.split("|"))
    else:
        order = S.block_order_from_onnx(args.onnx)
        if order is None:
            raise RuntimeError("could not detect observation block order from the policy ONNX")
    rm = ReferenceManager(PL.Planner(args.planner), al)
    if getattr(args, "no_cycle", False):
        rm.cycle = None      # forward walk from the planner, as the robot's refsrc runs it

    ob = S.ObsBuilder(al, types.SimpleNamespace(dof=np.zeros((2, 31)), T=2,
                      rot=np.array([[1.0, 0, 0, 0]] * 2)), def_il, rel_bias=rel_bias, order=order)
    ob.ref_pos_g1, ob.ref_vel_g1 = rm.pos, rm.vel
    ob.clip = types.SimpleNamespace(T=0, rot=None, trans=None, dof=None)  # (re)pointed each control step

    # stand the robot at its RL default pose
    data.qpos[:3] = [0.0, 0.0, PL.DEFAULT_HEIGHT]
    data.qpos[3:7] = [1.0, 0, 0, 0]
    data.qpos[7:] = def_il[S.IL2MJ]
    mujoco.mj_forward(model, data)
    return dict(model=model, data=data, pol=pol, ob=ob, rm=rm, order=order,
                off_il=off_il, scale_il=scale_il, al=al, lo=lo, hi=hi)


def control_step(ctx, t, cmd, prev_a, changed=False):
    import mujoco
    m, data, pol, ob, rm = ctx["model"], ctx["data"], ctx["pol"], ctx["ob"], ctx["rm"]
    al, off_il, scale_il, lo, hi = ctx["al"], ctx["off_il"], ctx["scale_il"], ctx["lo"], ctx["hi"]
    rm.heading += cmd.turn * S.CTRL_DT                 # integrate yaw-rate command into a heading
    rm.update(t, cmd, changed)
    ob.ref_pos_g1, ob.ref_vel_g1 = rm.pos, rm.vel      # point at the current plan
    ob.clip.rot, ob.clip.T = rot_quats(rm.rot, rm.heading), rm.T   # rotate maob heading target
    obs = ob.build(t, data.qpos[7:][S.MJ2IL], data.qvel[6:][S.MJ2IL],
                   data.qpos[3:7].copy(), data.qvel[3:6].copy(), prev_a)
    a = pol(obs)
    q_des = off_il + al.act_back(a) * scale_il
    data.ctrl[:] = np.clip(q_des[S.IL2MJ], lo, hi)
    for _ in range(int(round(S.CTRL_DT / S.PHYS_DT))):
        mujoco.mj_step(m, data)
    return a


def run_interactive(args):
    """Own GLFW window + polled key STATE (hold to drive).  Not mujoco.viewer:
    that only gives discrete key events and its native bindings collide with
    WASD.  Here we poll held keys each frame, so hold-W-to-walk / hold-A-to-turn
    feels continuous, and no key touches a viewer built-in."""
    import glfw
    import mujoco

    ctx = setup(args)
    m, data, rm = ctx["model"], ctx["data"], ctx["rm"]
    cmd = Command(args.speed)
    rm.init(data.qpos.copy(), cmd)

    if not glfw.init():
        raise RuntimeError("glfw.init() failed — is there a display? (interactive needs a screen)")
    win = glfw.create_window(1280, 720, "X2 planner teleop", None, None)
    if not win:
        glfw.terminate()
        raise RuntimeError("no window — run on a machine with a display (or use --script headless)")
    glfw.make_context_current(win)
    glfw.swap_interval(1)
    cam, opt, scn = mujoco.MjvCamera(), mujoco.MjvOption(), mujoco.MjvScene(m, maxgeom=10000)
    con = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
    cam.azimuth, cam.elevation, cam.distance = 140, -15, 3.5
    cam.lookat[:] = data.qpos[:3]

    ms = {"x": 0.0, "y": 0.0, "L": False, "R": False}

    def on_button(w, b, act, mods):
        ms["L"] = glfw.get_mouse_button(w, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        ms["R"] = glfw.get_mouse_button(w, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS

    def on_move(w, x, y):
        dx, dy = x - ms["x"], y - ms["y"]
        ms["x"], ms["y"] = x, y
        if ms["L"]:
            mujoco.mjv_moveCamera(m, mujoco.mjtMouse.mjMOUSE_ROTATE_V, dx / 500, dy / 500, scn, cam)
        elif ms["R"]:
            mujoco.mjv_moveCamera(m, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, dy / 500, scn, cam)

    def on_scroll(w, xo, yo):
        mujoco.mjv_moveCamera(m, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * yo, scn, cam)

    glfw.set_mouse_button_callback(win, on_button)
    glfw.set_cursor_pos_callback(win, on_move)
    glfw.set_scroll_callback(win, on_scroll)

    def held(k):
        return glfw.get_key(win, k) == glfw.PRESS

    turn_rate = math.radians(45)          # yaw rate while A/D held (rad/s)
    prev_plus = prev_minus = False
    prev_a = np.zeros(29)
    t, last = 0, None
    print("KEYS:  hold W walk | S back | Shift+W run | A/D turn | +/- speed | mouse drag/scroll camera | Esc quit")
    while not glfw.window_should_close(win) and not held(glfw.KEY_ESCAPE):
        wall = time.time()
        fwd = held(glfw.KEY_W)
        back = held(glfw.KEY_S)
        run = held(glfw.KEY_LEFT_SHIFT) or held(glfw.KEY_RIGHT_SHIFT)
        if fwd:
            cmd.mode, cmd.move = (3 if run else 1), np.array([1.0, 0.0, 0.0])
        elif back:
            cmd.mode, cmd.move = 2, np.array([-1.0, 0.0, 0.0])       # walk backward
        else:
            cmd.mode = 0                                             # stand
        cmd.turn = (turn_rate if held(glfw.KEY_A) else 0.0) - (turn_rate if held(glfw.KEY_D) else 0.0)
        plus = held(glfw.KEY_EQUAL) or held(glfw.KEY_KP_ADD)
        minus = held(glfw.KEY_MINUS) or held(glfw.KEY_KP_SUBTRACT)
        if plus and not prev_plus:
            cmd.speed = (2.0 if cmd.speed < 0 else cmd.speed) + 0.5
        if minus and not prev_minus:
            cmd.speed = max(-1.0, (2.0 if cmd.speed < 0 else cmd.speed) - 0.5)
        prev_plus, prev_minus = plus, minus

        cur = (cmd.mode, round(cmd.speed, 2), cmd.move[0])   # turn is a yaw-rate, not a plan change
        changed = cur != last
        last = cur
        prev_a = control_step(ctx, t, cmd, prev_a, changed)
        if data.qpos[2] < 0.4:
            setup_reset(ctx, cmd)
            prev_a = np.zeros(29)
            t = 0
        else:
            t += 1

        cam.lookat[:] = data.qpos[:3]
        fw, fh = glfw.get_framebuffer_size(win)
        vp = mujoco.MjrRect(0, 0, fw, fh)
        mujoco.mjv_updateScene(m, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        mujoco.mjr_render(vp, scn, con)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, vp,
                           f"{cmd.label()}   xy=({data.qpos[0]:+.1f},{data.qpos[1]:+.1f})",
                           "W walk | S back | Shift+W run | A/D turn | +/- speed | Esc quit", con)
        glfw.swap_buffers(win)
        glfw.poll_events()
        lag = S.CTRL_DT - (time.time() - wall)
        if lag > 0:
            time.sleep(lag)
    glfw.terminate()


def setup_reset(ctx, cmd):
    import mujoco
    m, data, rm = ctx["model"], ctx["data"], ctx["rm"]
    data.qpos[:3] = [data.qpos[0], data.qpos[1], PL.DEFAULT_HEIGHT]
    data.qpos[3:7] = [1.0, 0, 0, 0]
    data.qvel[:] = 0
    mujoco.mj_forward(m, data)
    cmd.mode = 0
    rm.heading = 0.0
    rm.init(data.qpos.copy(), cmd)


def run_script(args):
    """Headless validation: scripted command timeline -> mp4."""
    import mujoco
    import cv2
    os.environ.setdefault("MUJOCO_GL", "egl")
    ctx = setup(args)
    m, data = ctx["model"], ctx["data"]
    cmd = Command()
    ctx["rm"].init(data.qpos.copy(), cmd)
    # (start_sec, mode, speed, turn_rate rad/s)
    schedule = [(0, 0, -1, 0), (1.0, 1, -1, 0), (4.5, 1, -1, 0.5),
                (8.0, 3, -1, 0), (11.5, 1, -1, 0), (14.0, 0, -1, 0)]
    total = int(args.seconds * 50)
    vw = cv2.VideoWriter(args.render, cv2.VideoWriter_fourcc(*"mp4v"), 50, (720, 540))
    rend = mujoco.Renderer(m, 540, 720)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 3.5, -15, 140
    prev_a = np.zeros(29)
    fell = False
    for t in range(total):
        sec = t / 50.0
        before = (cmd.mode, cmd.speed)
        for (s, md, sp, tn) in schedule:
            if sec >= s:
                cmd.mode, cmd.speed, cmd.turn = md, sp, tn
        changed = (cmd.mode, cmd.speed) != before
        prev_a = control_step(ctx, t, cmd, prev_a, changed)
        if not fell and data.qpos[2] < 0.4:
            fell = True
            print(f"  [fell] t={sec:.2f}s h={data.qpos[2]:.2f}")
        if t % 100 == 0:
            print(f"  t={sec:5.2f}s {cmd.label()}  h={data.qpos[2]:.3f} "
                  f"xy=({data.qpos[0]:+.2f},{data.qpos[1]:+.2f}) yaw={math.degrees(yaw_of(data.qpos[3:7])):+.0f}")
        cam.lookat[:] = data.qpos[:3]
        rend.update_scene(data, cam)
        vw.write(cv2.cvtColor(rend.render(), cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"video: {args.render}  ({'FELL' if fell else 'stayed up'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=DEFAULT_POLICY)
    ap.add_argument("--block_order", default="",
                    help="override ONNX observation order; normally auto-detected")
    ap.add_argument("--planner", default=PL.PLANNER)
    ap.add_argument("--speed", type=float, default=-1.0,
                    help="initial planner target speed in m/s (-1 uses the mode default)")
    ap.add_argument("--no-cycle", dest="no_cycle", action="store_true",
                    help="forward walk from the planner instead of the human walk cycle; "
                         "matches what the robot's planner_refsrc_x2.py generates")
    ap.add_argument("--script", action="store_true", help="headless scripted self-test -> mp4")
    ap.add_argument("--seconds", type=float, default=16.0)
    ap.add_argument("--render", default="out/teleop_selftest.mp4")
    args = ap.parse_args()
    for label, path in (("V21 policy", args.onnx), ("SONIC planner", args.planner),
                        ("X2 MJCF", PL.X2_XML)):
        if not os.path.exists(path):
            raise SystemExit(f"missing {label}: {path}")
    if args.script:
        run_script(args)
    else:
        run_interactive(args)


if __name__ == "__main__":
    main()
