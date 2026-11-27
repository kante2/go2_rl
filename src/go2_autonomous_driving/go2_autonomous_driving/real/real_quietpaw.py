"""QuietPaw / QPace 48-input real Go2 controller; default is motor-free inference.

Both families append one noise-condition scalar to the 47 baseline inputs:
  quietpaw (e.g. model_9999): cost limit epsilon from --quietness 0..1 (1 = strict)
  qpace    (e.g. model_1000): normalized target dB from --target-db (saved [min_db, max_db])
Export first with scripts/prepare_go2_real_quietpaw.sh [--checkpoint PT]. Usually started by
scripts/run_go2_real_closed_loop.sh --model N, which picks this controller from the checkpoint.
The existing keyboard and measured-state viewer work with this controller too.
"""
import argparse
import math
from pathlib import Path
import signal
import sys

import numpy as np

from .real_control import PolicyMapping


BASE_OBSERVATIONS = ['base_ang_vel', 'projected_gravity', 'command', 'phase',
                     'joint_pos', 'joint_vel', 'actions']
QUIETPAW_OBSERVATIONS = BASE_OBSERVATIONS + ['noise_condition']


def cost_limit_from_quietness(quietness, condition):
    """The actor sees epsilon, not the user-facing quietness setting."""
    quietness = float(quietness)
    lower, upper = float(condition['min_cost_limit']), float(condition['max_cost_limit'])
    if not math.isfinite(quietness) or not 0 <= quietness <= 1:
        raise ValueError('quietness must be finite and in [0, 1]')
    if not all(math.isfinite(v) for v in (lower, upper)) or not 0 <= lower < upper <= 1:
        raise ValueError('Invalid saved QuietPaw cost limits; export the checkpoint again')
    return upper - quietness * (upper - lower)


def qpace_condition_from_target_db(target_db, condition):
    """QPace's actor sees (target_db - min_db) / (max_db - min_db), 0 = quietest trained target."""
    target_db = float(target_db)
    lower, upper = float(condition['min_db']), float(condition['max_db'])
    if not all(math.isfinite(v) for v in (lower, upper)) or not lower < upper:
        raise ValueError('Invalid saved QPace dB range; export the checkpoint again')
    if not math.isfinite(target_db) or not lower <= target_db <= upper:
        raise ValueError(f'target_db must be in the trained range [{lower:g}, {upper:g}] dB')
    return (target_db - lower) / (upper - lower)


def default_target_db(condition):
    """QPace play default: the cost reference level (baseline walk - 2.5 dB)."""
    return float(condition.get('reference_db', (condition['min_db'] + condition['max_db']) / 2))


CONDITION_FAMILIES = ('quietpaw', 'qpace')


class QuietPawMapping(PolicyMapping):
    """47 measured/base inputs followed by the saved noise condition (QuietPaw or QPace).

    Normalization is already embedded in the exported ONNX actor. Never
    normalize these raw observations twice, or append quietness / dB directly.
    """
    observation_size = 48

    def __init__(self, metadata, quietness=0.5, target_db=None):
        if (metadata.get('policy_family') not in CONDITION_FAMILIES
                or metadata.get('observation_size') != 48
                or metadata.get('observation_names') != QUIETPAW_OBSERVATIONS):
            raise ValueError('Expected exported QuietPaw/QPace 48-input metadata; run prepare_go2_real_quietpaw.sh')
        super().__init__({**metadata, 'observation_names': BASE_OBSERVATIONS})
        self.meta = metadata
        self.family = metadata['policy_family']
        condition = metadata['noise_condition']
        if self.family == 'quietpaw':
            self.quietness = float(quietness)
            self.cost_limit = self.condition = cost_limit_from_quietness(quietness, condition)
            self.description = f'quietness={self.quietness:g} cost_limit={self.cost_limit:g}'
        else:
            self.target_db = default_target_db(condition) if target_db is None else float(target_db)
            self.condition = qpace_condition_from_target_db(self.target_db, condition)
            self.description = f'target_db={self.target_db:g} condition={self.condition:.4f}'

    def observation(self, q_sdk, dq_sdk, quaternion, gyro, command, step, previous_action):
        base = super().observation(q_sdk, dq_sdk, quaternion, gyro, command, step, previous_action)
        return np.concatenate((base, np.array([[self.condition]], dtype=np.float32)), axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--policy-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--quietness', type=float, default=0.5,
                        help='QuietPaw only: 0 loose noise constraint, 1 strict; fixed for this run')
    parser.add_argument('--target-db', type=float,
                        help='QPace only: target robot noise [dB], in the trained range (default reference_db)')
    parser.add_argument('--execute', action='store_true', help='Motor output only after interactive ARM')
    parser.add_argument('--seconds', type=float)
    parser.add_argument('--fault-log', type=Path)
    # Same velocity caps as real_closed_loop (run_go2_real_closed_loop.sh passes them for every model).
    parser.add_argument('--speed', dest='max_speed', type=float, help='Max forward/backward speed [m/s]')
    parser.add_argument('--lateral-speed', dest='max_lateral_speed', type=float, help='Max lateral speed [m/s]')
    parser.add_argument('--yaw-rate', dest='max_yaw_rate', type=float, help='Max yaw rate [rad/s]')
    args = parser.parse_args()
    if not math.isfinite(args.quietness) or not 0 <= args.quietness <= 1:
        parser.error('--quietness must be finite and in [0, 1]')

    # Keep pure mapping/export checks usable without importing ROS or creating nodes.
    from .real_closed_loop import ClosedLoopBackend
    from .real_robot import run

    class QuietPawBackend(ClosedLoopBackend):
        def __init__(self, robot, session, input_name):
            super().__init__(robot, session, input_name)
            self.config.update(policy_family=self.mapping.family, noise_condition_input=self.mapping.condition)
            print(f'{self.mapping.family.upper()}: 48 inputs, {self.mapping.description}; '
                  f'checkpoint normalization included.', flush=True)

        def status(self):
            return super().status() + ' ' + self.mapping.description

    def interrupted(*unused):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    try:
        # Reuse ARM, freshness/CRC/tilt/tracking checks, keyboard timeout, target
        # ramps, 500 Hz motor output, emergency stop and final damping unchanged.
        run(args, backend_type=QuietPawBackend, requires_navigation=False, send_telemetry=True,
            mapping_factory=lambda metadata: QuietPawMapping(metadata, args.quietness, args.target_db))
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f'STOPPED: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
