# Unitree RL Mjlab


## ✳️ 概述

Unitree RL Mjlab 是一个基于 [mjlab](https://github.com/mujocolab/mjlab.git) 构建的强化学习项目，
使用 MuJoCo 作为物理仿真后端，当前仅支持 Unitree Go2 四足机器人。

Mjlab 结合了 [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 的成熟高层 API 与 
[MuJoCo](https://github.com/google-deepmind/mujoco_warp) 的高精度物理引擎，
为强化学习机器人研究与 Sim-to-Real（仿真到实机） 部署提供了一个轻量化、模块化的框架。

<div align="center">

| <div align="center">  MuJoCo </div>                                                                                                                                           | <div align="center"> Physical </div>                                                                                                                                               |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <div style="width:250px; height:150px; overflow:hidden;"><img src="doc/gif/go2-velocity.gif" style="width:100%; height:100%; object-fit:cover; object-position:center;"></div> | <div style="width:250px; height:150px; overflow:hidden;"><img src="doc/gif/go2-velocity-real.gif" style="width:100%; height:100%; object-fit:cover; object-position:center;"></div> |

</div>


## 📦 安装配置

安装和配置步骤请参考 [setup.md](doc/setup_zh.md)


## 🔁 流程概览

使用强化学习实现机器人运动控制的基本流程如下：

`训练` → `仿真验证` → `仿真到实机`

- **训练**: 在 MuJoCo 模拟环境中让机器人与环境交互，并通过奖励函数最大化学习策略。
- **仿真验证**: 加载训练好的策略进行回放，验证策略行为是否符合预期。
- **仿真到实机**: 将策略部署到物理机器人上，实现真实环境中的运动控制。


### 按训练配置顺序阅读源码

```text
src/locomotion_rl/go2_baseline/
├── registry.py                 注册平地和复杂地形任务
├── step_01_environment/        场景、机器人、动作、指令和随机化
│   ├── scene.py, actions.py, commands.py, events.py
│   └── base_env.py, go2_env.py
├── step_02_learning/           观测、奖励、终止条件和课程学习
│   ├── observations.py, rewards.py, terminations.py, curriculum.py, metrics.py
│   └── mdp/                   各配置项使用的计算函数
├── step_03_training/           PPO 配置、训练调度和模型保存
│   └── ppo.py, runner.py, train.py
└── step_04_evaluation/         加载并回放训练后的策略
    └── play.py
```

文件夹编号表示阅读和配置流程的顺序。实际训练时，观测 → 策略/动作 → 物理仿真 →
奖励与下一次观测会持续循环，每轮采集经验后进行 PPO 更新。
机器人模型及控制常量仍位于 `src/assets/robots/unitree_go2/`；
`scripts/train.py` 和 `scripts/play.py` 保留原有命令接口，分别调用第 03、04 步。
详细的韩文流程及配置说明见 [kante_.md](kante_.md)。

## 🛠️ 使用指南

### 1. 速度跟踪训练

运行以下命令进行速度跟踪训练：

```bash
python scripts/train.py Unitree-Go2-Flat --env.scene.num-envs=4096
```

如需快速检查执行流程，可使用 8 个环境运行 3 次 PPO 迭代，每个环境每轮采集 8 步：

```bash
python scripts/train.py Unitree-Go2-Flat \
  --env.scene.num-envs 8 \
  --agent.max-iterations 3 \
  --agent.num-steps-per-env 8 \
  --agent.save-interval 1 \
  --agent.logger tensorboard \
  --agent.run-name refactor_smoke
```

该命令用于检查训练、检查点保存及导出流程，不代表策略已学会稳定行走。
结果保存在 `logs/rsl_rl/go2_velocity/*_refactor_smoke/` 下。

多 GPU 训练：使用 --gpu-ids 扩展到多块 GPU：

```bash
python scripts/train.py Unitree-Go2-Flat \
  --gpu-ids 0 1 \
  --env.scene.num-envs=4096
```

- 第一个参数(如 Unitree-Go2-Flat)为必选参数，确定要启用的训练环境。可选：
  - Unitree-Go2-Flat
  - Unitree-Go2-Rough

> [!NOTE]
> 更多有关详细说明，请参阅 mjlab 文档
> [mjlab documentation](https://mujocolab.github.io/mjlab/index.html).

#### ⚙️  参数说明
- `--env.scene`: 仿真场景配置，包括环境数量（num_envs）、物理仿真步长、地面类型、重力、随机扰动等参数。
- `--env.observations`: 观测空间配置，控制训练时输入到策略网络的状态信息，如关节位置、速度、IMU等内容。
- `--env.rewards`: 奖励函数配置，定义每步训练时的优化目标。
- `--env.commands`: 控制命令配置，用于生成训练时随机或指定的速度 / 姿态 / 动作指令。
- `--env.terminations`: 终止条件配置，定义训练 episode 的结束条件。
- `--agent.seed`: 训练随机种子，用于结果复现，不同 seed 会导致策略略有差异。
- `--agent.resume`: 是否从上次中断的 checkpoint 继续训练。 设置为 True 时，会自动加载最近一次保存的 .pt 模型文件。
- `--agent.policy`: 策略网络结构配置，例如 MLP 层数、隐藏维度、激活函数等。
- `--agent.algorithm`: 强化学习算法配置。可设置优化超参数，如学习率、批量大小、GAE λ 等。

**默认保存训练结果**：`logs/rsl_rl/go2_velocity/<date_time>/model_<iteration>.pt`

### 2. 仿真验证

如果想要在 MuJoCo 中查看训练效果，可以运行以下命令：

查看速度跟踪训练效果：
```bash
python scripts/play.py Unitree-Go2-Flat --checkpoint-file=logs/rsl_rl/go2_velocity/2026-xx-xx_xx-xx-xx/model_xx.pt
```

**说明**：

- 训练时在每次保存模型时会同步导出 policy.onnx 文件在同层目录下，可用于实物部署。

**效果**：

| Go2                              |
|----------------------------------|
| ![go2](doc/gif/go2-velocity.gif) |

### 3. 实物部署

实物部署前先确保主机安装了下列通信工具：
- [cyclonedds](https://github.com/eclipse-cyclonedds/cyclonedds.git)
- [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2.git)

<div style="margin-left: 20px;">

#### 3.1 启动机器人
将机器人在吊装状态下启动，并等待机器人进入 `零力矩模式`

#### 3.2 进入调试模式
确保机器人处于 `零力矩模式` 的情况下，按下遥控器的 `L2+R2`组合键；此时机器人会进入`调试模式`, `调试模式`下机器人关节处于阻尼状态。

#### 3.3 连接机器人
使用网线连接电脑与机器人网口，并修改网络配置如下：
- 地址：`192.168.123.222`
- 子网掩码：`255.255.255.0`

然后使用 `ifconfig` 命令查看与机器人连接的网卡名称，记录后用于启动参数。

#### 3.4 编译
以 Unitree Go2 速度控制为例。
将策略文件（`policy.onnx`）放入`deploy/robots/go2/config/policy/velocity/v0/exported` 下，然后执行：

```bash
cd deploy/robots/go2
mkdir build && cd build
cmake .. && make
```

#### 3.5 部署

## 3.5.1 仿真部署

在实物部署前，建议使用[unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)进行仿真部署，防止实物机器人出现异常动作。本框架已将其集成。

编译unitree_mujoco：

```bash
cd simulate
mkdir build && cd build
cmake .. && make -j8
```

启动仿真器(注意此处需连接上手柄才能启动)：

```bash
./simulate/build/unitree_mujoco
```

`simulate/config.yaml` 默认使用 Go2。

启动仿真控制程序：

```bash
cd deploy/robots/go2/build
./go2_ctrl --network=lo
```

## 3.5.2 实物部署

启动实物控制程序：

```bash
cd deploy/robots/go2/build
./go2_ctrl --network=enp5s0
```

**参数说明**：
- `network`: 连接机器人网卡名称，仿真部署使用 `lo`，实物机器人如 `enp5s0`(可使用 `ifconfig` 指令查看)

</div>

**实物效果**：

| Go2                                                    |
|--------------------------------------------------------|
| <img src="doc/gif/go2-velocity-real.gif" width="300"/> |


## 🎉  致谢

本仓库开发离不开以下开源项目的支持与贡献，特此感谢：

- [mjlab](https://github.com/mujocolab/mjlab.git): 构建训练与运行代码的基础。
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git): 强化学习算法实现。
- [mujoco_warp](https://github.com/google-deepmind/mujoco_warp.git): 提供 GPU 加速渲染与仿真接口。
- [mujoco](https://github.com/google-deepmind/mujoco.git): 提供强大仿真功能。

