# Unitree RL Mjlab


## ✳️ Overview
Unitree RL Mjlab is a reinforcement learning project built upon the
[mjlab](https://github.com/mujocolab/mjlab.git), using MuJoCo as its 
physics simulation backend, currently supporting only the Unitree Go2 quadruped.

Mjlab combines [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s proven API
with best-in-class [MuJoCo](https://github.com/google-deepmind/mujoco_warp)
physics to provide lightweight, modular abstractions for RL robotics research
and sim-to-real deployment.

<div align="center">

| <div align="center">  MuJoCo </div>                                                                                                                                           | <div align="center"> Physical </div>                                                                                                                                               |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <div style="width:250px; height:150px; overflow:hidden;"><img src="doc/gif/go2-velocity.gif" style="width:100%; height:100%; object-fit:cover; object-position:center;"></div> | <div style="width:250px; height:150px; overflow:hidden;"><img src="doc/gif/go2-velocity-real.gif" style="width:100%; height:100%; object-fit:cover; object-position:center;"></div> |

</div>


## 📦 Installation and Configuration

Please refer to [setup.md](doc/setup_en.md) for installation and configuration steps.

For the Go2 Docker environment, GPU training, simulation keyboard control and real-robot
closed-loop control with a web viewer, see [the Docker guide](docker/README.md).


## 🔁 Process Overview

The basic workflow for using reinforcement learning to achieve motion control is:

`Train` → `Play` → `Sim2Real`

- **Train**: The agent interacts with the MuJoCo simulation and optimizes policies through reward maximization.
- **Play**: Replay trained policies to verify expected behavior.
- **Sim2Real**: Deploy trained policies to physical Unitree robots for real-world execution.


### Source layout: follow the training setup

```text
src/tasks/go2/
├── registry.py                 Register the Flat/Rough tasks
├── step_01_environment/        Scene, robot, actions, commands, randomization
│   ├── scene.py, actions.py, commands.py, events.py
│   └── base_env.py, go2_env.py
├── step_02_learning/           Observations, rewards, episode rules, curriculum
│   ├── observations.py, rewards.py, terminations.py, curriculum.py, metrics.py
│   └── mdp/                   Functions that compute the configured terms
├── step_03_training/           PPO settings, training orchestration, saving
│   └── ppo.py, runner.py, train.py
└── step_04_evaluation/         Load and replay trained policies
    └── play.py
```

The folder numbers show the order for reading and configuring the training pipeline.
During training, observations → policy/actions → physics → rewards and next observations
repeat in a loop, with PPO updates after each rollout. Robot models and control constants
remain in `src/assets/robots/unitree_go2/`; the existing `scripts/train.py` and
`scripts/play.py` commands are thin entry points into steps 03 and 04.
See [kante_.md](kante_.md) for the detailed Korean guide and configuration map.

## 🛠️ Usage Guide

### 1. Velocity Tracking Training

Run the following command to train a velocity tracking policy:

```bash
python scripts/train.py Unitree-Go2-Flat --env.scene.num-envs=4096
```

For a short execution check (8 environments, 3 PPO iterations, 8 rollout steps per
environment), run:

```bash
python scripts/train.py Unitree-Go2-Flat \
  --env.scene.num-envs 8 \
  --agent.max-iterations 3 \
  --agent.num-steps-per-env 8 \
  --agent.save-interval 1 \
  --agent.logger tensorboard \
  --agent.run-name refactor_smoke
```

This checks the training and checkpoint/export paths; it does not establish walking
performance. Results are saved under `logs/rsl_rl/go2_velocity/*_refactor_smoke/`.

Multi-GPU Training: Scale to multiple GPUs using --gpu-ids:

```bash
python scripts/train.py Unitree-Go2-Flat \
  --gpu-ids 0 1 \
  --env.scene.num-envs=4096
```

- The first argument (e.g., Unitree-Go2-Flat) specifies the training task.
Available velocity tracking tasks:
  - Unitree-Go2-Flat
  - Unitree-Go2-Rough

> [!NOTE]
> For more details, refer to the mjlab documentation:
> [mjlab documentation](https://mujocolab.github.io/mjlab/index.html).

#### ⚙️  Parameter Description
- `--env.scene`: simulation scene configuration (e.g., num_envs, dt, ground type, gravity, disturbances)
- `--env.observations`: observation space configuration (e.g., joint state, IMU, commands, etc.)
- `--env.rewards`: reward terms used for policy optimization
- `--env.commands`: task commands (e.g., velocity, pose, or motion targets)
- `--env.terminations`: termination conditions for each episode
- `--agent.seed`: random seed for reproducibility
- `--agent.resume`: resume from the last saved checkpoint when enabled
- `--agent.policy`: policy network architecture configuration
- `--agent.algorithm`: reinforcement learning algorithm configuration (PPO, hyperparameters, etc.)

**Training results are stored at**：`logs/rsl_rl/go2_velocity/<date_time>/model_<iteration>.pt`

### 2. Simulation Validation

To visualize policy behavior in MuJoCo:

Velocity tracking:
```bash
python scripts/play.py Unitree-Go2-Flat --checkpoint-file=logs/rsl_rl/go2_velocity/2026-xx-xx_xx-xx-xx/model_xx.pt
```

**Note**：

- During training, policy.onnx and policy.onnx.data are also exported for deployment onto physical robots.

**Visualization**：

| Go2                              |
|----------------------------------|
| ![go2](doc/gif/go2-velocity.gif) |

### 3. Real Deployment

Before deployment, install the required communication tools:
- [cyclonedds](https://github.com/eclipse-cyclonedds/cyclonedds.git)
- [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2.git)

<div style="margin-left: 20px;">

#### 3.1 Power On the Robot
Start the robot in suspended state and wait until it enters `zero-torque` mode.

#### 3.2 Enable Debug Mode
While in `zero-torque` mode, press `L2 + R2` on the controller. The robot will enter `debug mode` with joint damping enabled.

#### 3.3 Connect to the Robot
Connect your PC to the robot via Ethernet. Configure the network as:
- Address：`192.168.123.222`
- Netmask：`255.255.255.0`

Use `ifconfig` to determine the Ethernet device name for deployment.

#### 3.4 Compilation

Example: Unitree Go2 velocity control.
Place `policy.onnx` and `policy.onnx.data` into: `deploy/robots/go2/config/policy/velocity/v0/exported`.
Then compile:

```bash
cd deploy/robots/go2
mkdir build && cd build
cmake .. && make
```

#### 3.5 Deployment

## 3.5.1 Simulation Deployment

Before deploying on the real robot, it is recommended to perform simulation deployment using [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
to prevent abnormal behaviors on the physical robot. This framework has already integrated it.

Build unitree_mujoco：

```bash
cd simulate
mkdir build && cd build
cmake .. && make -j8
```

Launch the simulator (note that a gamepad must be connected):

```bash
./simulate/build/unitree_mujoco
```

Go2 is configured by default in `simulate/config.yaml`.

Launch the simulation control program:

```bash
cd deploy/robots/go2/build
./go2_ctrl --network=lo
```

## 3.5.2 Real-Robot Deployment

Launch the control program on the real robot:

```bash
cd deploy/robots/go2/build
./go2_ctrl --network=enp5s0
```

**Arguments**：
- `network`: The network interface used to connect to the robot. Use `lo` for simulation deployment, and `enp5s0` for the real robot(You can check it using the `ifconfig` command) 

</div>

**Deployment Results**：

| Go2                                                    |
|--------------------------------------------------------|
| <img src="doc/gif/go2-velocity-real.gif" width="300"/> |


## 🎉  Acknowledgements

This project would not be possible without the contributions of the following repositories:

- [mjlab](https://github.com/mujocolab/mjlab.git): training and execution framework
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git): reinforcement learning algorithm implementation
- [mujoco_warp](https://github.com/google-deepmind/mujoco_warp.git): GPU-accelerated rendering and simulation interface
- [mujoco](https://github.com/google-deepmind/mujoco.git): high-fidelity rigid-body physics engine
