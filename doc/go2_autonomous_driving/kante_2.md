# 1. 학습한 모델 보행 성능 평가  

docker start unitree-rl-mjlab

docker exec -it -w /workspace/unitree_rl_mjlab unitree-rl-mjlab \
  python scripts/play.py Unitree-Go2-Flat \
  --checkpoint-file /workspace/unitree_rl_mjlab/src/go2_autonomous_driving/model_pt/model_10000.pt \
  --num-envs 1 \
  --viewer viser



+---------------------------------+
|         Base Environment        |
+------------------------+--------+
| Property               | Value  |
+------------------------+--------+
| Number of environments | 1      |
| Environment device     | cuda:0 |
| Environment seed       | None   |
| Physics step-size      | 0.005  |
| Environment step-size  | 0.02   |
+------------------------+--------+

[INFO] <EventManager> contains 2 active terms.
+-------------------------------------+
| Active Event Terms in Mode: 'reset' |
+--------+----------------------------+
| Index  | Name                       |
+--------+----------------------------+
|   0    | reset_base                 |
|   1    | reset_robot_joints         |
|   2    | randomize_terrain          |
+--------+----------------------------+
+---------------------------------------+
| Active Event Terms in Mode: 'startup' |
+-----------+---------------------------+
|   Index   | Name                      |
+-----------+---------------------------+
|     0     | foot_friction             |
|     1     | encoder_bias              |
|     2     | base_com                  |
+-----------+---------------------------+
+-----------------------------+
| Domain Randomization Fields |
+-------+---------------------+
| Index | Field Name          |
+-------+---------------------+
|   0   | geom_friction       |
|   1   | body_ipos           |
|   2   | body_subtreemass    |
|   3   | dof_invweight0      |
|   4   | body_invweight0     |
|   5   | tendon_length0      |
|   6   | tendon_invweight0   |
+-------+---------------------+

[INFO] <CommandManager> contains 1 active terms.
+----------------------------------------+
|          Active Command Terms          |
+-------+-------+------------------------+
| Index | Name  |          Type          |
+-------+-------+------------------------+
|   0   | twist | UniformVelocityCommand |
+-------+-------+------------------------+

[INFO] <ActionManager> contains 1 active terms.
+---------------------------------+
| Active Action Terms (shape: 12) |
+-------+-----------+-------------+
| Index | Name      |   Dimension |
+-------+-----------+-------------+
|   0   | joint_pos |          12 |
+-------+-----------+-------------+

[INFO] <ObservationManager> contains 2 groups.
+-----------------------------------------------------------+
| Active Observation Terms in Group: 'actor' (shape: (47,)) |
+--------+------------------------+-------------------------+
| Index  | Name                   |          Shape          |
+--------+------------------------+-------------------------+
|   0    | base_ang_vel           |      (3,)  ← 1×(3,)     |
|   1    | projected_gravity      |      (3,)  ← 1×(3,)     |
|   2    | command                |      (3,)  ← 1×(3,)     |
|   3    | phase                  |      (2,)  ← 1×(2,)     |
|   4    | joint_pos              |     (12,)  ← 1×(12,)    |
|   5    | joint_vel              |     (12,)  ← 1×(12,)    |
|   6    | actions                |     (12,)  ← 1×(12,)    |
+--------+------------------------+-------------------------+
+------------------------------------------------------------+
| Active Observation Terms in Group: 'critic' (shape: (74,)) |
+--------+--------------------------+------------------------+
| Index  | Name                     |         Shape          |
+--------+--------------------------+------------------------+
|   0    | base_ang_vel             |     (3,)  ← 1×(3,)     |
|   1    | projected_gravity        |     (3,)  ← 1×(3,)     |
|   2    | command                  |     (3,)  ← 1×(3,)     |
|   3    | phase                    |     (2,)  ← 1×(2,)     |
|   4    | joint_pos                |    (12,)  ← 1×(12,)    |
|   5    | joint_vel                |    (12,)  ← 1×(12,)    |
|   6    | actions                  |    (12,)  ← 1×(12,)    |
|   7    | base_lin_vel             |     (3,)  ← 1×(3,)     |
|   8    | foot_height              |     (4,)  ← 1×(4,)     |
|   9    | foot_air_time            |     (4,)  ← 1×(4,)     |
|   10   | foot_contact             |     (4,)  ← 1×(4,)     |
|   11   | foot_contact_forces      |    (12,)  ← 1×(12,)    |
+--------+--------------------------+------------------------+

[INFO] <TerminationManager> contains 3 active terms.
+------------------------------------+
|      Active Termination Terms      |
+-------+-----------------+----------+
| Index | Name            | Time Out |
+-------+-----------------+----------+
|   0   | time_out        |   True   |
|   1   | fell_over       |  False   |
|   2   | illegal_contact |  False   |
+-------+-----------------+----------+

[INFO] <RewardManager> contains 15 active terms.
+-------------------------------------------+
|            Active Reward Terms            |
+-------+------------------------+----------+
| Index | Name                   |   Weight |
+-------+------------------------+----------+
|   0   | track_linear_velocity  |      1.0 |
|   1   | track_angular_velocity |      1.0 |
|   2   | body_orientation_l2    |     -1.0 |
|   3   | pose                   |      1.0 |
|   4   | body_ang_vel           |    -0.05 |
|   5   | angular_momentum       |   -0.025 |
|   6   | is_terminated          |   -200.0 |
|   7   | joint_acc_l2           | -2.5e-07 |
|   8   | joint_pos_limits       |    -10.0 |
|   9   | action_rate_l2         |    -0.05 |
|   10  | foot_gait              |      0.5 |
|   11  | foot_clearance         |     -1.0 |
|   12  | foot_slip              |    -0.25 |
|   13  | soft_landing           |   -0.001 |
|   14  | stand_still            |     -1.0 |
+-------+------------------------+----------+

[INFO] <NullCurriculumManager> (inactive)
[INFO] <MetricsManager> contains 1 active terms.
+-------------------------+
|   Active Metrics Terms  |
+-------+-----------------+
| Index | Name            |
+-------+-----------------+
|   0   | mean_action_acc |
+-------+-----------------+

Module mujoco_warp._src.io b0ffa0e load on device 'cuda:0' took 7.09 ms  (cached)
Module _tile_cholesky_factorize__locals__cholesky_factorize_bd98e5db 78076c6 load on device 'cuda:0' took 27.04 ms  (cached)
Module _tile_cholesky_solve__locals__cholesky_solve_7bd08b0a 54b9810 load on device 'cuda:0' took 26.52 ms  (cached)
--------------------------------------------------------------------------------
Resolved observation sets: 
         actor :  ('actor',)
         critic :  ('critic',)
--------------------------------------------------------------------------------
Actor Model: MLPModel(
  (obs_normalizer): EmpiricalNormalization()
  (distribution): GaussianDistribution()
  (mlp): MLP(
    (0): Linear(in_features=47, out_features=512, bias=True)
    (1): ELU(alpha=1.0)
    (2): Linear(in_features=512, out_features=256, bias=True)
    (3): ELU(alpha=1.0)
    (4): Linear(in_features=256, out_features=128, bias=True)
    (5): ELU(alpha=1.0)
    (6): Linear(in_features=128, out_features=12, bias=True)
  )
)
Critic Model: MLPModel(
  (obs_normalizer): EmpiricalNormalization()
  (mlp): MLP(
    (0): Linear(in_features=74, out_features=512, bias=True)
    (1): ELU(alpha=1.0)
    (2): Linear(in_features=512, out_features=256, bias=True)
    (3): ELU(alpha=1.0)
    (4): Linear(in_features=256, out_features=128, bias=True)
    (5): ELU(alpha=1.0)
    (6): Linear(in_features=128, out_features=1, bias=True)
  )
)
╭────── viser (listening *:8080) ───────╮
│             ╷                         │
│   HTTP      │ http://localhost:8080   │
│   Websocket │ ws://localhost:8080     │
│             ╵                         │
╰───────────────────────────────────────╯
(viser) Connection opened (0, 1 total), 230 persistent messages





# 2-----------------------------------------------------
# 키보드 컨트롤 결과

# 2-1 플래너 실행
cd /home/kante/unitree_rl_mjlab
source /opt/ros/jazzy/setup.bash
source src/go2_autonomous_driving/install/setup.bash
export ROS_DOMAIN_ID=42
ros2 launch go2_autonomous_driving simulation.launch.py

# 2-2 무주코/ 모델 실행
cd /home/kante/unitree_rl_mjlab
bash src/go2_autonomous_driving/scripts/run_simulation.sh

# 2-3 키보드 실행
cd /home/kante/unitree_rl_mjlab
source /opt/ros/jazzy/setup.bash
source src/go2_autonomous_driving/install/setup.bash
export ROS_DOMAIN_ID=42
ros2 run go2_autonomous_driving keyboard_waypointㄱ