"""LowState-only foot motion proxy, without estimating body translation."""
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'preprocessing'))
from go2_kinematics import Go2Kinematics, LEGS

FEATURE_SCHEMA = 'lowstate_rotated_relative_vertical_v1'
VELOCITY_DESCRIPTION = 'LowState-only rotated relative foot motion z: [R(omega cross p + J dq)]_z; body translation excluded'


def vertical_motion(low, model=None):
    """Nx35: force4, quaternion wxyz4, gyro3, q12, dq12 -> Nx4 m/s.

    This is NOT absolute world foot velocity: world body translation is absent.
    All four motor/force blocks use FR, FL, RR, RL order, as in training.
    """
    low = np.asarray(low, dtype=float)
    if low.ndim != 2 or low.shape[1] != 35 or not np.isfinite(low).all():
        raise ValueError('LowState must contain finite (N,35) values')
    quat = low[:, 4:8]
    if np.any(np.linalg.norm(quat, axis=1) < 1e-8):
        raise ValueError('Invalid LowState attitude quaternion')
    rotation = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
    model = model or Go2Kinematics()
    gyro, q, dq = low[:, 8:11], low[:, 11:23], low[:, 23:35]
    result = []
    for i, leg in enumerate(LEGS):
        p, jacobian = model.position_jacobian(leg, q[:, 3*i:3*i+3])
        relative = np.einsum('nij,nj->ni', jacobian, dq[:, 3*i:3*i+3])
        result.append(rotation.apply(relative + np.cross(gyro, p))[:, 2])
    return np.column_stack(result)


def frame_lowstate(frame):
    columns = ([f'ls_ff_{leg}' for leg in LEGS] + [f'ls_quaternion_{a}' for a in 'wxyz']
               + [f'ls_gyro_{a}' for a in 'xyz']
               + [f'ls_{group}_{leg}_{joint}' for group in ('q', 'dq') for leg in LEGS
                  for joint in ('hip', 'thigh', 'calf')])
    return frame[columns].to_numpy(float)
