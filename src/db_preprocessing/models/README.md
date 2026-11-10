# Go2 reference kinematics

`go2_description.urdf` is copied from the official Unitree model:

https://github.com/unitreerobotics/unitree_ros/blob/master/robots/go2_description/urdf/go2_description.urdf

The original BSD-3-Clause notice is in `LICENSE.unitree`. Mesh files are not
needed for forward kinematics and are not bundled. The source URDF remains
unchanged. Each event-export run records its SHA-256 in the diagnostics.

The user confirmed the recordings use a Unitree Go2. The calculation uses
FR/FL/RR/RL hip/thigh/calf SDK order and the URDF foot-frame origin. This is a
rigid point proxy for the foot; sole deformation and the instantaneous ground
contact point are not modelled. The IMU joint has identity rotation relative
to the base, so gyro axes are used as body axes. Sport body velocity is treated
as base-origin velocity; its exact estimator reference-point convention is
not independently calibrated.
