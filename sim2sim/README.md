# sim2sim — MuJoCo 部署验证

把 Sonic→X2 (Any2Any) 训好的策略拿到 MuJoCo 物理里闭环跑,验证 sim2sim 迁移。
详细复现记录见 `../BUGS_AND_FIXES.md` 的 "sim2sim MuJoCo 部署重建" 一节。

## 可移交的交互行走 Demo

这个入口只依赖 Python、MuJoCo 和 ONNX Runtime，不需要 Isaac Sim、
Isaac Lab、PyTorch 或训练数据。

```bash
cd Any2Any/sim2sim
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-walking.txt

# 按 models/README.md 放入两个 ONNX 文件后：
./run_walking_demo.sh --speed 0.5
```

按住 `W` 前进，`Shift+W` 跑，`S` 后退，`A/D` 转向，`+/-` 调速，
`Esc` 退出。当前 V21 step 12000 在 MuJoCo 中以 `--speed 0.5` 测得：
`W` 的 walk trajectory 实际约 `0.115 m/s`，`Shift+W` 的 run trajectory
实际约 `0.45 m/s`；这里的参数是 planner 目标，不是根速度闭环设定值。
程序会直接从 ONNX 图检测 observation block 顺序；不要为不同导出版本
硬编码 `cmf|maob|pro`。

运行所需的 X2 MJCF/meshes 已在 `../../GMR/assets/agibot_x2/`。策略和
planner 权重不进普通 Git；文件名、大小、校验值及模型许可证见
`models/README.md` 与 `licenses/`。

发送单个压缩包时，可在有模型文件的机器上执行：

```bash
./build_walking_bundle.sh /path/to/model_step_012000_g1.onnx \
  /path/to/planner_sonic.onnx
```

脚本会校验两个模型的 SHA-256，并生成包含代码、X2 meshes、模型和许可证的
`walking_demo_v21_s12000.tar.gz`。

## 当前目录

```
sim2sim/
  mujoco_sim2sim_x2.py     harness
  run_sim2sim.sh           一键管线
  README.md
  capture/                 V21 obs 真值，harness 默认从这里读取
  out/                     V21 planner / teleop 近期验证输出
  v21_s10000/demos/        V21 step 10000 回归对比
  v21_s12000/demos/        V21 step 12000 当前部署演示
```

`capture/` 含从 IsaacSim 捕获的干净真值 `constants.npz` / `gt_step0..2.npz` /
`rel_bias.npy`，让 harness 无需 IsaacSim 即可跑 `--validate` 和闭环渲染。
`--capture_dir` 不传时默认使用 `./capture`。

## 文件

- `mujoco_sim2sim_x2.py` — harness。`--validate` 是 obs 契约验证门(纯 CPU,逐块对
  IsaacSim 干净真值);默认闭环跑 MuJoCo 物理并渲染视频。obs 契约细节写在文件头 docstring。
  捕获探针在 `../../GR00T-WholeBodyControl/gear_sonic/train_agent_trl.py` 的 PLAY 块(ANY2ANY_CAPTURE)。
- `run_sim2sim.sh <run_dir> <version> [clip] [steps] [ckpt]` — 一键管线:merge LoRA →
  导 ONNX(需 IsaacSim)→ 捕获真值到 `<version>/capture` → 验证门 → 闭环渲染到 `<version>/<clip>.mp4`。
  `ckpt` 省略=用 run 的 last.pt(会快照);也可传具名快照(如 `snap_v15_iter5499.pt`)。

## 用法

```bash
# 从 GR00T-WholeBodyControl/ 执行，使用当前 V21 step 12000:
bash ../Any2Any/sim2sim/run_sim2sim.sh \
  logs_rl/TRL_X2_Track/v21 v21_s12000 walk1_subject1 1500 \
  logs_rl/TRL_X2_Track/v21/model_step_012000.pt

# 手动验证门（纯 CPU；--capture_dir 不传时使用 ./capture）:
python mujoco_sim2sim_x2.py --validate \
  --onnx ../../GR00T-WholeBodyControl/logs_rl/TRL_X2_Track/v21/exported/model_step_012000_g1.onnx

# 单条 clip 闭环 + 视频(手动指定版本 capture 与输出):
MUJOCO_GL=egl python mujoco_sim2sim_x2.py --clip walk1_subject1 --steps 1500 \
    --onnx ../../GR00T-WholeBodyControl/logs_rl/TRL_X2_Track/v21/exported/model_step_012000_g1.onnx \
    --capture_dir capture --render out/walk1_subject1.mp4 --ref_ghost
```

## 结果快照

- **V21 step 12000** — 当前部署版本；演示见 `v21_s12000/demos/`，planner / teleop 闭环结果见 `out/`。
- **V21 step 10000** — 仅作同版本回归对比，不是默认部署入口。

验证门标准:proprio 精确 0.0000、maob ~1e-5、cmf ~0.05(30→50fps 插值底噪)、端到端动作 ~0.1(FSQ 容忍)。

⚠️ ONNX 输入块序**每次导出可能变**(导出代码用 `list(set())`);验证门 A 段误差 >1 即错序,
按 `mujoco_sim2sim_x2.py` docstring 的排列穷举重定 `ObsBuilder.build` 拼接顺序。
