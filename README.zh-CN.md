<div align="center">

# Sonic2AgiX2

### 把 SONIC 的全身控制能力迁移到智元 X2。

**Unitree G1 → AgiBot X2 Ultra · 运动学对齐 + LoRA · MuJoCo 闭环验证**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Paper: Any2Any](https://img.shields.io/badge/Paper-Any2Any-b31b1b.svg)](https://arxiv.org/abs/2605.23733)

[English](README.md) · **简体中文** · [运行指南](sim2sim/README.md) · [迁移避坑清单](docs/CHECKLIST.md)

<img src="media/x2_walk_to_bow.gif" alt="智元 X2 行走后鞠躬：左侧为 MuJoCo 闭环策略，右侧为运动参考" width="900">

**从行走到鞠躬。** 左：MuJoCo 中的闭环控制；右：运动学参考。

</div>

**一个在 G1 上预训练的全身控制策略，如何适应另一台人形机器人？**

Sonic2AgiX2 将 NVIDIA **GEAR-SONIC** 从 29 自由度的 Unitree G1 迁移到 **31 自由度的智元 AgiBot X2 Ultra**。项目独立实现了 [Any2Any](https://arxiv.org/abs/2605.23733) 的迁移思路：先对齐两台机器人的运动学与策略接口，再用 LoRA 适配 actor 动力学解码器和 critic。

这里包含围绕迁移构建的工程链路：**人体动作重定向 → 策略适配 → MuJoCo 验证 → 真机部署工具**，也保留了关节映射、观测布局与数据筛选中的实际排错记录。

> **发布范围：** 本仓库是接入外部 SONIC / GMR 工作区的源码扩展。模型权重、机器人资产、动作数据集和评估捕获文件均未打包，需要自行准备；当前版本不是克隆后即可复现训练的一键运行包。

## 值得关注的部分

- **跨本体复用策略：** G1↔X2 关节空间对齐，处理关节顺序、腕部语义配对和零位偏移；通过 LoRA 适配动力学相关模块。
- **人体动作到机器人：** 提供 AMASS、LAFAN、OMOMO、100STYLE 批处理入口、X2 标定配置和逐片段质量检查。
- **看得见的闭环验证：** MuJoCo + ONNX 推理、策略与参考并排渲染，以及对照 Isaac Lab 捕获真值的观测检查。
- **键盘交互行走：** planner 驱动的 WASD 控制、走跑切换，以及可选的人体步态循环参考。
- **部署工具链：** 模型打包、数值测试向量、dry-run 桥接、手柄工具和 C++ 参考动作源示例。
- **可复用的排错经验：** 记录索引错位、动作缩放、软限位和观测接口错误如何影响迁移结果。

## 已记录的结果

项目原有发布说明记录的 **v21** 全身跟踪结果：

| 评估集 | 已报告成功率 |
|---|---:|
| OMOMO | **97.0%** |
| LAFAN | **72.9%** |

以上为项目已有报告值，本次源码发布未附对应 checkpoint、完整评估协议和原始日志，也未重新运行评估。请勿把这些数字视为可直接复现的基准或跨方法比较。首页动图展示的是 MuJoCo 仿真，不是真机试验。

## 如何开始

### 准备所需资源

| 目标 | 需要自行准备 |
|---|---|
| 阅读迁移实现 | 本仓库源码 |
| MuJoCo 交互演示 | Python 依赖、兼容的 X2 policy ONNX、SONIC planner ONNX、X2 MJCF 及 meshes |
| 观测验证 / 指定动作回放 | 匹配的 Isaac Lab 捕获文件、重定向动作片段，以及相应策略和机器人资产 |
| 人体动作重定向 | 打补丁后的 GMR、X2 资产、输入格式所需的数据集和人体模型 |
| 策略训练与导出 | 上游 SONIC trainer、Isaac Sim / Isaac Lab、NVIDIA GPU、基础权重、动作库和 X2 训练集成 |

权重、数据集、SMPL-X 模型和机器人描述请从各自提供方获取，并遵循对应许可。

### 建立工作区

多个脚本依赖同级目录布局：

```text
workspace/
├── Sonic2AgiX2/
├── GR00T-WholeBodyControl/
└── GMR/
```

在工作区目录执行：

```bash
git clone https://github.com/TaiE64/Sonic2AgiX2.git
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git
git clone https://github.com/YanjieZe/GMR.git

export SONIC2X2_ROOT="$PWD/Sonic2AgiX2"
export GROOT_ROOT="$PWD/GR00T-WholeBodyControl"
export PYTHONPATH="$SONIC2X2_ROOT/retarget:$SONIC2X2_ROOT/any2any_train${PYTHONPATH:+:$PYTHONPATH}"
```

`PYTHONPATH` 用于暴露源码发布中移入子目录的模块。`GROOT_ROOT` 在这里方便 shell 命令引用；并非所有脚本都会读取它，部分脚本仍需调整原工作区路径。

### 按目标选择入口

- **先跑仿真：** 按 [MuJoCo 运行指南](sim2sim/README.md) 指定模型和资产路径。交互推理不依赖 Isaac Sim 或 PyTorch。
- **处理动作数据：** 先阅读 [GMR overlay 说明](gmr_x2_overlay/README.md)，对照记录的上游版本应用补丁，再选择 [retarget/](retarget/) 下的批处理入口。
- **接入训练：** 按上游说明安装 trainer，将 [x2.py](gear_sonic_x2_overlay/x2.py) 放入其 `gear_sonic/envs/manager_env/robots/` 目录，再集成 alignment wrapper、LoRA、动作库和实验配置。仅复制机器人配置不能复现 v21；当前发布未包含完整可移植训练配方。

完整步骤和源码入口见 [英文 README](README.md#getting-started)。

## 代码导航

| 目录 | 内容 |
|---|---|
| [any2any_train/](any2any_train/) | 运动学对齐、LoRA、checkpoint 合并、动作库转换 |
| [retarget/](retarget/) | 重定向、关节映射、标定、逐片段 QC |
| [sim2sim/](sim2sim/) | MuJoCo 验证、ONNX 回放、planner、键盘遥操作 |
| [gmr_x2_overlay/](gmr_x2_overlay/) | GMR 补丁、上游版本、X2 IK 配置 |
| [gear_sonic_x2_overlay/](gear_sonic_x2_overlay/) | SONIC 使用的 X2 机器人与跟踪配置 |
| [deploy_x2/](deploy_x2/) | 部署打包、测试向量、桥接与参考动作源示例 |
| [data_manifests/](data_manifests/) | 训练数据筛选清单，仅含片段名称 |

## 做跨机器人迁移，建议先读这三份记录

- [迁移 Checklist](docs/CHECKLIST.md)：逐项检查索引、语义配对、观测契约、动作缩放和软限位。
- [Bugs & Fixes](docs/BUGS_AND_FIXES.md)：实际症状、根因、修复证据，以及被后续实验推翻的判断。
- [重定向笔记](docs/RETARGET_NOTES.md)：求解器设计与历史验证约束。

这些笔记保留了原工作区路径与版本背景，接入当前仓库时请以现有代码和运行指南为准。

## 参与项目

欢迎提交可复现的 bug、安装流程改进、评估脚本和新的机器人映射。提 issue 时请附上 commit、环境、执行命令和最小日志，不要上传第三方受限权重或数据集。

如果这些代码和排错记录对你的研究有帮助，欢迎 **Star**，也欢迎在 [Issues](https://github.com/TaiE64/Sonic2AgiX2/issues) 分享迁移结果和遇到的问题。

## 致谢与许可

感谢 [Any2Any](https://arxiv.org/abs/2605.23733)、NVIDIA [GR00T-WholeBodyControl / GEAR-SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) 和 [GMR](https://github.com/YanjieZe/GMR)。本项目为独立实现，不代表上述团队的官方发布。

本仓库代码采用 [Apache-2.0](LICENSE)，保留相关上游声明；外部代码、模型权重、数据集与机器人资产分别遵循其自身许可。
