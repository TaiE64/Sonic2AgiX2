"""Serve X2REFv1 reference frames from the SONIC planner (runs ON the robot).

Replaces the workstation as the reference source for the deployed tracking
controller: instead of streaming a human-motion reference from a laptop, the
planner runs locally and turns a velocity command into the same 50 Hz frames.
The robot becomes self-contained -- nothing outside this host is in the loop.

Wire protocol (must match X2PolicyPipeline::LiveRxLoop exactly):
    8-byte magic "X2REFv1\\0", then fixed 256-byte frames of
    u64 seq | 29 f32 joint pos (G1-IL) | 29 f32 joint vel | 4 f32 root quat wxyz
Native byte order; both ends are little-endian aarch64/x86.

PACING IS PART OF THE CONTRACT.  In live mode the controller pins its playhead
to `newest frame - X2_LIVE_FUTURE_FRAMES` (x2_policy_pipeline.cc:310), so the
frame ARRIVAL RATE is the playback rate.  Emitting a whole planner segment at
once would jump the playhead ~73 frames.  Frames therefore leave here on a
50 Hz monotonic clock, and the plan is grown ahead of the emitter in bulk.

The planner call costs ~200 ms on this robot's CPU and yields ~1.47 s of
motion, so a plan kept ~1.2 s ahead of the emitter absorbs the latency.

Reference implementation: sim2sim/planner_teleop_x2.py (ReferenceManager).
Kept free of mujoco/torch so it runs against the robot's stock python3.
"""

from __future__ import annotations

import argparse
import math
import os
import socket
import struct
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import planner_lib as PL  # noqa: E402

# il[j] = mj[G1_MJ2IL[j]] -- copied from Any2Any/kinematic_alignment.py so this
# file stays importable on the robot without the training-side dependencies.
G1_MJ2IL = np.array([
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5,
    11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
])

MAGIC = b"X2REFv1\0"
FRAME_FMT = "<Q29f29f4f"
FRAME_SIZE = struct.calcsize(FRAME_FMT)
assert FRAME_SIZE == 256, FRAME_SIZE

CTRL_DT = 0.02                       # 50 Hz
CONTEXT_FRAMES = 4                   # planner frames spent reconstructing its input context

# Forward walk tracks a looped human gait cycle (AMASS B3 retargeted to X2),
# NOT the planner: the planner's own slow_walk is left/right asymmetric at the
# source and comes out jerky and barely progressing on hardware. Baked offline
# into G1-MJ frames by bake_walk_cycle so this file needs no mujoco/alignment.
HUMAN_WALK_CYCLE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "human_walk_cycle_g1mj.npz")
CYCLE_SWITCH_BLEND = 10              # 50Hz frames to morph when entering the cycle

# Command clamps, matching the 11600 controller's external /x2rpf/cmd_vel gate.
VX_MAX, YAW_MAX = 0.6, 0.6
# Below this the command is IDLE (stand). Guards against a drifting stick or a
# near-zero cmd_vel producing a full-speed gait reference; it also keeps the
# commanded speed out of the band below the policy's trained minimum.
VX_IDLE = 0.10
# X2_Locomotion was trained on planner references over walk 0.2-0.6 m/s, so the
# stick maps into that band instead of the teleop's single fixed speed.
WALK_MIN, WALK_MAX = 0.2, 0.6


class Command:
    """Velocity command -> planner mode/speed + a synthesized yaw rate.

    Turning is NOT a planner input: the planner always walks straight (facing
    FORWARD) and the turn is produced by rotating the reference root
    orientation, which the policy closes as a heading error.
    """

    def __init__(self, vx=0.0, yaw=0.0):
        self.set(vx, yaw)

    def set(self, vx, yaw):
        self.vx = float(np.clip(vx, -VX_MAX, VX_MAX))
        self.turn = float(np.clip(yaw, -YAW_MAX, YAW_MAX))
        if abs(self.vx) < VX_IDLE:
            self.mode, self.speed = 0, -1.0                  # IDLE (stand)
        else:                                                # SLOW_WALK, speed from stick travel
            self.mode = 1
            self.speed = float(np.clip(abs(self.vx), WALK_MIN, WALK_MAX))
        self.move = np.array([1.0 if self.vx >= 0 else -1.0, 0.0, 0.0])

    def key(self):
        """Identity used to detect a command change worth re-planning for.
        Speed is quantized: it now varies continuously with the stick, and every
        micro-change would otherwise truncate and re-plan on every tick."""
        return (self.mode, round(self.speed, 1), self.move[0])

    def label(self):
        return (f"mode={self.mode}({PL.MODE_NAMES.get(self.mode, '?')}) "
                f"vx={self.vx:+.2f} yaw={math.degrees(self.turn):+.0f}deg/s")


class ReferenceManager:
    """Online planner plan, ported from planner_teleop_x2.ReferenceManager.

    Holds the plan as a G1-MJ world-frame trajectory at 50 Hz and re-seeds each
    replan from the plan's OWN continuation, not from the robot's live state:
    seeding from the robot restarts the gait from stand every tick and marches
    in place.  Human-walk-cycle mode is intentionally omitted here -- it needs
    the X2->G1 kinematic alignment, which is training-side.
    """

    def __init__(self, planner, seed=1234):
        self.pl = planner
        self.seed = seed
        self.plan = None                 # (N,36) G1-MJ world @50Hz
        self.pos = self.vel = self.rot = None
        self.T = 0
        # Frame indices are ABSOLUTE: `base` is the absolute index of plan[0],
        # so compaction never shifts the emitter's cursor out from under it.
        self.base = 0
        # Planning runs off the emit thread (a replan blocks ~200 ms on this
        # robot, and the emitter must not miss its 50 Hz tick). Every mutation
        # swaps whole arrays under this lock; readers snapshot and then index
        # freely, since nothing is mutated in place.
        self.lock = threading.Lock()
        self.cycle = None                # (M,36) G1-MJ human walk cycle @50Hz, +X heading
        self.cycle_stride = 0.0
        self._plan_phase = []            # per plan frame: cycle index, or -1 for planner frames
        self._load_cycle()

    def _load_cycle(self):
        if not os.path.exists(HUMAN_WALK_CYCLE):
            print(f"[refsrc] WARNING: no walk cycle at {HUMAN_WALK_CYCLE}; forward walk "
                  "falls back to the planner gait (jerky on hardware)", flush=True)
            return
        z = np.load(HUMAN_WALK_CYCLE)
        self.cycle = z["cycle"].astype(np.float64)
        self.cycle_stride = float(z["stride"])
        m = len(self.cycle)
        print(f"[refsrc] walk cycle: {m} frames ({m / 50.0:.2f}s), stride "
              f"{self.cycle_stride:.3f}m -> {self.cycle_stride / (m / 50.0):.3f} m/s", flush=True)

    @staticmethod
    def _z_rotate(yaw, quats):
        """Compose R_z(yaw) with wxyz quaternion rows."""
        w, z = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
        qw, qx, qy, qz = np.asarray(quats).T
        return np.stack([w * qw - z * qz, w * qx - z * qy,
                         w * qy + z * qx, w * qz + z * qw], axis=1)

    def _cycle_dx(self, a):
        """Cycle-local displacement from frame a to frame (a+1) % M."""
        m = len(self.cycle)
        b = (a + 1) % m
        d = self.cycle[b, :2] - self.cycle[a, :2]
        if b == 0:
            d = d + np.array([self.cycle_stride, 0.0])
        return d

    def _extend_cycle(self):
        """Append one gait cycle at the plan tip: joints and root orientation
        loop, xy integrates the cycle's own step displacement rotated into the
        tip heading. Port of planner_teleop_x2.ReferenceManager._extend_cycle."""
        plan = self.plan
        gen = len(plan) - 1
        tip = plan[gen]
        switching = self._plan_phase[gen] < 0
        m = len(self.cycle)
        start = 0 if switching else (self._plan_phase[gen] + 1) % m
        idx = [(start + k) % m for k in range(m)]

        qw, qx, qy, qz = tip[3:7]
        tip_yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        c = self.cycle[start]
        cyc_yaw = math.atan2(2 * (c[3] * c[6] + c[4] * c[5]), 1 - 2 * (c[5] ** 2 + c[6] ** 2))
        spin = tip_yaw - cyc_yaw
        rot_mat = np.array([[math.cos(spin), -math.sin(spin)],
                            [math.sin(spin), math.cos(spin)]])

        seg = self.cycle[idx].copy()
        seg[:, 3:7] = self._z_rotate(spin, seg[:, 3:7])
        prev_idx = (start - 1) % m
        xy = tip[:2].copy()
        for k, _ in enumerate(idx):
            a = prev_idx if k == 0 else idx[k - 1]
            xy = xy + rot_mat @ self._cycle_dx(a)
            seg[k, :2] = xy
        if switching:                       # morph out of whatever pose the tip holds
            for j in range(min(CYCLE_SWITCH_BLEND, m)):
                alpha = (j + 1) / (CYCLE_SWITCH_BLEND + 1)
                seg[j, 7:] = (1 - alpha) * tip[7:] + alpha * seg[j, 7:]
                seg[j, 2] = (1 - alpha) * tip[2] + alpha * seg[j, 2]
                q0, q1 = tip[3:7], seg[j, 3:7]
                if np.dot(q0, q1) < 0:
                    q1 = -q1
                q = (1 - alpha) * q0 + alpha * q1
                seg[j, 3:7] = q / np.linalg.norm(q)
        with self.lock:
            self.plan = np.concatenate([self.plan, seg], axis=0)
            self._plan_phase.extend(idx)
            self._derive()
        return True

    def _derive(self):
        self.pos = self.plan[:, 7:][:, G1_MJ2IL]                 # G1-IL joints
        self.vel = np.diff(self.pos, axis=0, append=self.pos[-1:]) * 50.0
        self.rot = self.plan[:, 3:7]                             # world root quat
        self.T = len(self.plan)

    def snapshot(self):
        """(pos, vel, rot, base, T) consistent with each other."""
        with self.lock:
            return self.pos, self.vel, self.rot, self.base, self.T

    def init(self, cmd=None):
        """Seed the plan with a STANDING segment, whatever the current command.

        The planner always generates this first segment, so seeding it with a
        walk would put ~1 s of planner gait in front of the walk cycle on every
        connect/reset -- the jerky gait, exactly where it hurts most. Standing
        also means a controller that arms right after connecting tracks a still
        reference instead of stepping off immediately; _extend then switches to
        the cycle (or the planner, for backward) with its own entry blend.
        """
        del cmd                                     # deliberately not used
        idle = Command(0.0, 0.0)
        q30 = self.pl.run(PL.standing_context(np.zeros(29)), idle.mode, idle.move,
                          PL.FORWARD, idle.speed, -1.0, self.seed)
        if len(q30) == 0 or np.isnan(q30).any():
            raise RuntimeError("planner returned no frames / NaNs on init")
        with self.lock:
            self.plan = PL.resample_50hz(q30)
            self._plan_phase = [-1] * len(self.plan)
            self.base = 0
            self._derive()

    def _extend(self, cmd):
        """Append one segment at the plan tip (chained_rollout style).

        Only the planner thread calls this, so reading `plan` needs no lock;
        the ~200 ms planner call stays outside the lock and only the array swap
        is held, so the emitter never waits on inference.
        """
        # Forward walk only: the cycle is a +X gait, so backward still needs the
        # planner (which reverses via `move`).
        if cmd.mode == 1 and cmd.move[0] > 0 and self.cycle is not None:
            return self._extend_cycle()
        plan = self.plan
        gen = len(plan) - 1
        ctx = PL.context_from_motion(plan, max(0, gen - 5))
        q30 = self.pl.run(ctx, cmd.mode, cmd.move, PL.FORWARD, cmd.speed, -1.0, self.seed)
        if len(q30) < CONTEXT_FRAMES + 2 or np.isnan(q30).any():
            return False
        # Drop the planner's lossy reconstruction of its own context: splicing
        # it in steps every joint at the seam, and the policy hitches when that
        # step enters its future window.
        continuation30 = np.concatenate([plan[gen:gen + 1], q30[CONTEXT_FRAMES:]], axis=0)
        seg = PL.resample_50hz(continuation30)
        seg[:, :2] += plan[gen, :2] - seg[0, :2]                 # xy re-anchor to the tip
        with self.lock:
            # `plan` may have been truncated meanwhile only by this same thread,
            # so appending to the live array is safe.
            self.plan = np.concatenate([self.plan, seg[1:]], axis=0)
            self._plan_phase.extend([-1] * (len(seg) - 1))
            self._derive()
        return True

    def update(self, emitted, cmd, changed, ahead=60):
        """Keep the plan `ahead` frames past the emitter (absolute index),
        re-planning from just past the emitter when the command changes."""
        if changed:
            keep = emitted + 12 - self.base
            if 0 < keep < len(self.plan):
                with self.lock:
                    self.plan = self.plan[:keep]
                    self._plan_phase = self._plan_phase[:keep]
                    self._derive()
        guard = 0
        while self.base + self.T < emitted + ahead and guard < 5:
            if not self._extend(cmd):
                break
            guard += 1

    def compact(self, emitted, keep_behind=100):
        """Bound memory on hour-long sessions. Absolute indices are unaffected."""
        drop = emitted - keep_behind - self.base
        if drop <= 0 or self.T < 500:
            return
        with self.lock:
            self.plan = self.plan[drop:]
            self._plan_phase = self._plan_phase[drop:]
            self.base += drop
            self._derive()


def z_rotate(yaw, q):
    """Rz(yaw) (x) q, wxyz."""
    w, z = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    qw, qx, qy, qz = q
    return np.array([w * qw - z * qz, w * qx - z * qy, w * qy + z * qx, w * qz + z * qw])


class CommandSource:
    """Fixed command, optionally driven by ROS /x2rpf/cmd_vel or the PS5 sticks.

    The gamepad drives only the reference VELOCITY here. Arming, damping and
    passive stay with the controller's own /joy handler -- this process has no
    command path to the robot at all, so it can never be the thing that moves it.
    """

    def __init__(self, vx, yaw, source, timeout=0.5, deadzone=0.12):
        self.cmd = Command(vx, yaw)
        self.source = source                 # "fixed" | "cmd_vel" | "joy"
        self.timeout = timeout
        self.deadzone = deadzone
        self.node = self.executor = None
        self._last_rx = 0.0
        self._rx = None
        if source in ("cmd_vel", "joy"):
            self._start_ros()

    def _dz(self, v):
        """Deadzone, rescaled so the live range still reaches full travel."""
        if abs(v) < self.deadzone:
            return 0.0
        return (abs(v) - self.deadzone) / (1.0 - self.deadzone) * (1.0 if v > 0 else -1.0)

    def _start_ros(self):
        import rclpy
        from rclpy.node import Node
        from rclpy.executors import SingleThreadedExecutor

        rclpy.init(args=None)
        self.node = Node("x2rpf_planner_refsrc")

        if self.source == "cmd_vel":
            from geometry_msgs.msg import Twist

            def on_cmd(msg):
                self._rx = (msg.linear.x, msg.angular.z)
                self._last_rx = time.monotonic()

            self.node.create_subscription(Twist, "/x2rpf/cmd_vel", on_cmd, 10)
            what = "/x2rpf/cmd_vel"
        else:
            from sensor_msgs.msg import Joy

            def on_joy(msg):
                if len(msg.axes) < 2:
                    return
                # Left stick, matching the 11600 launcher's convention:
                # vertical = walk, horizontal = turn (ROS joy: left is +).
                self._rx = (self._dz(msg.axes[1]) * VX_MAX,
                            self._dz(msg.axes[0]) * YAW_MAX)
                self._last_rx = time.monotonic()

            self.node.create_subscription(Joy, "/joy", on_joy, 10)
            what = "/joy left stick (up=walk, left=turn)"

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        # Spin on a dedicated thread. Spinning from whichever thread happens to
        # call poll() invalidates the executor's wait set ("context is not
        # valid"), which silently killed the planner thread.
        threading.Thread(target=self.executor.spin, daemon=True).start()
        print(f"[refsrc] commands from {what} "
              f"(clamps vx {VX_MAX}, yaw {YAW_MAX}, {self.timeout}s timeout -> zero)", flush=True)

    def poll(self):
        """Returns (cmd, changed)."""
        before = self.cmd.key()
        if self.node is not None:
            if self._rx is not None and time.monotonic() - self._last_rx < self.timeout:
                self.cmd.set(*self._rx)
            else:
                self.cmd.set(0.0, 0.0)           # stale/never -> stop, same as the C++ gate
        return self.cmd, self.cmd.key() != before

    def shutdown(self):
        if self.node is not None:
            import rclpy
            self.executor.remove_node(self.node)
            self.node.destroy_node()
            rclpy.shutdown()


def plan_loop(rm, src, args, stop, cursor):
    """Own the planner. Runs off the emit thread: one planner call costs ~200 ms
    on the robot and would otherwise blow the emitter's 50 Hz tick (measured:
    41 Hz with 400 ms stalls when this ran inline)."""
    while not stop.is_set():
        try:
            cmd, changed = src.poll()
            rm.update(cursor[0], cmd, changed, ahead=args.ahead)
            rm.compact(cursor[0])
        except Exception as exc:                 # a dead planner must not wedge the emitter
            print(f"[refsrc] planner failed: {exc}", flush=True)
            stop.set()
            return
        time.sleep(0.005)


def serve(conn, rm, src, args):
    """Emit 50 Hz frames until the peer goes away."""
    conn.sendall(MAGIC)
    seq = 0
    cursor = [0]                # absolute index of the next frame to emit
    heading = 0.0
    starved = 0
    late = 0
    stop = threading.Event()
    planner = threading.Thread(target=plan_loop, args=(rm, src, args, stop, cursor), daemon=True)
    planner.start()

    next_due = last_log = time.monotonic()
    try:
        while not stop.is_set():
            now = time.monotonic()
            if now < next_due:
                time.sleep(min(next_due - now, 0.002))
                continue
            if now - next_due > 0.1:             # fell far behind: resync, do not burst
                late += 1
                next_due = now
            next_due += CTRL_DT

            pos, vel, rot, base, T = rm.snapshot()
            i = cursor[0] - base
            if i >= T - 1:                       # planner starved; hold (controller freezes t)
                starved += 1
                continue

            cmd = src.cmd
            heading += cmd.turn * CTRL_DT        # heading advances with the frame timeline
            # Lead the heading so the frame carries the yaw command it will have
            # when it reaches the controller's playhead, `future` frames from now.
            q = z_rotate(heading + cmd.turn * args.future * CTRL_DT, rot[i])
            frame = struct.pack(FRAME_FMT, seq, *pos[i].astype(np.float32),
                                *vel[i].astype(np.float32), *q.astype(np.float32))
            try:
                conn.sendall(frame)
            except OSError as exc:
                print(f"[refsrc] peer gone: {exc}", flush=True)
                return
            seq += 1
            cursor[0] += 1

            if now - last_log >= 5.0:
                print(f"[refsrc] seq={seq} plan_ahead={base + T - cursor[0]}f "
                      f"({(base + T - cursor[0]) * CTRL_DT:.2f}s) starved={starved} "
                      f"late_resyncs={late} heading={math.degrees(heading):+.0f}deg "
                      f"{cmd.label()}", flush=True)
                last_log = now
    finally:
        stop.set()
        planner.join(timeout=3.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", default=os.environ.get(
        "SONIC_PLANNER_ONNX", PL.PLANNER), help="planner_sonic.onnx")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default: loopback only)")
    ap.add_argument("--port", type=int, default=5602)
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--ros", action="store_true", help="take commands from /x2rpf/cmd_vel")
    ap.add_argument("--joy", action="store_true",
                    help="take commands from the PS5 left stick on /joy "
                         "(arming/damping stay with the controller)")
    ap.add_argument("--deadzone", type=float, default=0.18)
    ap.add_argument("--ahead", type=int, default=60,
                    help="plan frames to keep past the emitter (60 = 1.2s)")
    ap.add_argument("--future", type=int, default=25,
                    help="controller's X2_LIVE_FUTURE_FRAMES; used for heading lead")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--planner-threads", type=int, default=2,
                    help="cap planner ONNX threads so its ~200ms bursts do not "
                         "starve the 50Hz control policy sharing this CPU (0 = all cores)")
    ap.add_argument("--dump", default="", help="write the first N emitted frames to this .npz and exit")
    ap.add_argument("--dump-frames", type=int, default=500)
    args = ap.parse_args()

    print(f"[refsrc] loading planner: {args.planner}", flush=True)
    t0 = time.time()
    rm = ReferenceManager(PL.Planner(args.planner, intra_threads=args.planner_threads),
                          seed=args.seed)
    print(f"[refsrc] planner loaded in {time.time() - t0:.1f}s", flush=True)

    source = "joy" if args.joy else ("cmd_vel" if args.ros else "fixed")
    src = CommandSource(args.vx, args.yaw, source, deadzone=args.deadzone)
    rm.init(src.cmd)
    print(f"[refsrc] initial plan {rm.T} frames ({rm.T * CTRL_DT:.2f}s), {src.cmd.label()}", flush=True)

    if args.dump:
        dump(rm, src, args)
        return

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f"[refsrc] serving X2REFv1 on {args.host}:{args.port}", flush=True)
    try:
        while True:
            conn, peer = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[refsrc] client {peer}", flush=True)
            try:
                serve(conn, rm, src, args)
            finally:
                conn.close()
            rm.init(src.cmd)          # fresh plan for the next session
            print("[refsrc] client gone; plan reset", flush=True)
    except KeyboardInterrupt:
        print("[refsrc] stopping", flush=True)
    finally:
        srv.close()
        src.shutdown()


def dump(rm, src, args):
    """Offline: generate frames without a peer, for numeric comparison."""
    pos, vel, rot = [], [], []
    heading = 0.0
    for i in range(args.dump_frames):
        cmd, changed = src.poll()
        heading += cmd.turn * CTRL_DT
        rm.update(i, cmd, changed, ahead=args.ahead)
        if i >= rm.T - 1:
            break
        pos.append(rm.pos[i].copy())
        vel.append(rm.vel[i].copy())
        rot.append(z_rotate(heading + cmd.turn * args.future * CTRL_DT, rm.rot[i]))
    np.savez(args.dump, pos=np.array(pos), vel=np.array(vel), rot=np.array(rot),
             vx=src.cmd.vx, yaw=src.cmd.turn, seed=args.seed)
    print(f"[refsrc] wrote {len(pos)} frames -> {args.dump}", flush=True)


if __name__ == "__main__":
    main()
