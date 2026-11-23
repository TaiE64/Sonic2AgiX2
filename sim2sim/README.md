# MuJoCo validation and keyboard teleoperation

[Project overview](../README.md) · [中文项目介绍](../README.zh-CN.md)

Run a compatible Sonic2AgiX2 policy in MuJoCo, compare it with a motion reference, or drive it with the SONIC locomotion planner. Inference does not require Isaac Sim or PyTorch. Training, ONNX export, and generation of Isaac Lab ground-truth captures remain separate steps.

## Required assets

This source checkout includes the scripts and [runtime_constants.json](runtime_constants.json), but **does not include**:

- An X2 policy ONNX or the SONIC planner ONNX.
- The X2 MJCF and its referenced mesh files.
- Retargeted motion clips, `human_walk_cycle.npz`, or Isaac Lab `capture/` files.
- The `models/README.md`, `licenses/`, or recorded demo directories referenced by older internal instructions.

Use a policy compatible with the included alignment and runtime constants. A policy trained with different joint ordering, gains, or observation conventions needs its matching configuration. Obtain third-party assets under their providers' terms.

## Interactive demo

The commands below assume a Linux machine with a working graphical display and compatible assets already available. Replace every `/path/to/` value. From the repository root:

```bash
export SONIC2X2_ROOT="$PWD"
export PYTHONPATH="$SONIC2X2_ROOT/retarget:$SONIC2X2_ROOT/any2any_train${PYTHONPATH:+:$PYTHONPATH}"
export X2_MJCF="/path/to/agibot_x2/x2_mocap.xml"

python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r sim2sim/requirements-walking.txt

bash sim2sim/run_walking_demo.sh \
  --onnx /path/to/x2_policy.onnx \
  --planner /path/to/planner_sonic.onnx \
  --speed 0.5
```

The shell wrapper uses `python3` from the active environment; set `DEMO_PYTHON` to select another interpreter. Supply `--onnx` explicitly: the script's default points to an author's external `X2_Locomotion` checkpoint.

| Input | Action |
|---|---|
| Hold `W` | Walk forward |
| Hold `Shift+W` | Run forward |
| Hold `S` | Walk backward |
| Hold `A` / `D` | Turn left / right |
| Release movement keys | Stand |
| `+` / `-` | Adjust planner target speed |
| Mouse drag / scroll | Orbit / zoom |
| `Esc` | Quit |

`--speed` sets a **planner target**, not a guaranteed measured robot velocity. When an externally prepared `human_walk_cycle.npz` is present in this directory, forward walking can use that cycle. Without it, the planner supplies the reference; `--no-cycle` explicitly selects planner-based walking.

The teleop code normally detects observation-block order from the ONNX graph. Do not assume that every policy export uses the same order.

## Headless teleop recording

The scripted recording path also imports OpenCV. On a Linux host with EGL rendering available:

```bash
python -m pip install opencv-python-headless
mkdir -p /path/to/output

MUJOCO_GL=egl python sim2sim/planner_teleop_x2.py \
  --onnx /path/to/x2_policy.onnx \
  --planner /path/to/planner_sonic.onnx \
  --script --seconds 16 \
  --render /path/to/output/teleop.mp4
```

Use the Python entry point directly here: `run_walking_demo.sh` intentionally unsets `MUJOCO_GL` for interactive use.

## Validate the observation contract

`--validate` compares rebuilt observations and ONNX actions against clean Isaac Lab captures. It requires `gt_step0.npz`, `gt_step1.npz`, `gt_step2.npz`, matching runtime constants, and the LAFAN motion library used for the capture. These files must come from your matching training/export environment.

```bash
# The validator looks for motion_lib_lafan_x2/robot beneath this directory.
export X2_MOTION_DATA="/path/to/motion_data"

python sim2sim/mujoco_sim2sim_x2.py \
  --validate \
  --capture_dir /path/to/matching/capture \
  --onnx /path/to/x2_policy.onnx
```

The validator reports network parity, proprioception, future-motion observations, and end-to-end action differences. It also writes `rel_bias.npy` into the capture directory. The shipped JSON constants support playback but do not replace ground-truth observations for this check.

Historical error levels and observation-layout details are recorded in [Bugs and Fixes](../docs/BUGS_AND_FIXES.md). Interpret them with the checkpoint, sampling rate, and export version that produced them; they are not universal tolerances.

## Replay a motion with a reference ghost

With OpenCV installed, the output directory created, and the same environment and matching assets configured:

```bash
MUJOCO_GL=egl python sim2sim/mujoco_sim2sim_x2.py \
  --onnx /path/to/x2_policy.onnx \
  --capture_dir /path/to/matching/capture \
  --motion_path /path/to/retargeted_clip.pkl \
  --steps 1500 \
  --render /path/to/output/motion.mp4 \
  --ref_ghost
```

## Script map and portability

| File | Purpose |
|---|---|
| [mujoco_sim2sim_x2.py](mujoco_sim2sim_x2.py) | Observation validation, closed-loop simulation, reference rendering |
| [planner_teleop_x2.py](planner_teleop_x2.py) | Interactive or scripted planner-driven control |
| [planner_lib.py](planner_lib.py) | Planner inference, resampling, and trajectory continuation |
| [generate_locomotion_references.py](generate_locomotion_references.py) | Locomotion reference generation |
| [planner_refsrc_x2.py](planner_refsrc_x2.py) | Reference source for the robot deployment path |
| [onnx_block_order.py](onnx_block_order.py) | Observation-layout detection from exported ONNX |
| [run_sim2sim.sh](run_sim2sim.sh) | Historical merge → export → capture → validate → render recipe |
| [build_walking_bundle.sh](build_walking_bundle.sh) | Historical workspace-specific packaging script |

`run_sim2sim.sh` contains an author-specific interpreter path, experiment names, and capture hooks that must exist in your trainer. Adapt it before use.

`build_walking_bundle.sh` expects a root-level `kinematic_alignment.py`, a local `licenses/` directory, external robot assets, and two exact model checksums. Those assumptions do not match a fresh source checkout. Treat it as a packaging reference, not an out-of-the-box release command; only package external assets where their licenses permit it.
