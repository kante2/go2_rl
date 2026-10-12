"""QuietPaw 48-input real Go2 controller; default is motor-free inference.

Export first with scripts/prepare_go2_real_quietpaw.sh. Run on the host with
scripts/run_go2_real_quietpaw.sh [--quietness 0.5] [--execute]. The existing
keyboard and measured-state viewer work with this controller too.
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


class QuietPawMapping(PolicyMapping):
    """47 measured/base inputs followed by the saved cost-limit condition.

    Normalization is already embedded in the exported ONNX actor. Never
    normalize these raw observations twice, or append quietness directly.
    """
    observation_size = 48

    def __init__(self, metadata, quietness=0.5):
        if (metadata.get('policy_family') != 'quietpaw'
                or metadata.get('observation_size') != 48
                or metadata.get('observation_names') != QUIETPAW_OBSERVATIONS):
            raise ValueError('Expected exported QuietPaw 48-input metadata; run prepare_go2_real_quietpaw.sh')
        super().__init__({**metadata, 'observation_names': BASE_OBSERVATIONS})
        self.meta = metadata
        self.quietness = float(quietness)
        self.cost_limit = cost_limit_from_quietness(quietness, metadata['noise_condition'])

    def observation(self, q_sdk, dq_sdk, quaternion, gyro, command, step, previous_action):
        base = super().observation(q_sdk, dq_sdk, quaternion, gyro, command, step, previous_action)
        return np.concatenate((base, np.array([[self.cost_limit]], dtype=np.float32)), axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--policy-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--quietness', type=float, default=0.5,
                        help='0: loose noise constraint, 1: strict; fixed for this run')
    parser.add_argument('--execute', action='store_true', help='Motor output only after interactive ARM')
    parser.add_argument('--seconds', type=float)
    parser.add_argument('--fault-log', type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.quietness) or not 0 <= args.quietness <= 1:
        parser.error('--quietness must be finite and in [0, 1]')

    # Keep pure mapping/export checks usable without importing ROS or creating nodes.
    from .real_closed_loop import ClosedLoopBackend
    from .real_robot import run

    class QuietPawBackend(ClosedLoopBackend):
        def __init__(self, robot, session, input_name):
            super().__init__(robot, session, input_name)
            self.config.update(quietness=self.mapping.quietness, cost_limit=self.mapping.cost_limit)
            print(f'QUIETPAW: 48 inputs, quietness={self.mapping.quietness:g}, '
                  f'cost_limit={self.mapping.cost_limit:g}; checkpoint normalization included.', flush=True)

        def status(self):
            return (super().status() + f' quietness={self.mapping.quietness:g}'
                    f' cost_limit={self.mapping.cost_limit:g}')

    def interrupted(*unused):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    try:
        # Reuse ARM, freshness/CRC/tilt/tracking checks, keyboard timeout, target
        # ramps, 500 Hz motor output, emergency stop and final damping unchanged.
        run(args, backend_type=QuietPawBackend, requires_navigation=False, send_telemetry=True,
            mapping_factory=lambda metadata: QuietPawMapping(metadata, args.quietness))
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f'STOPPED: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
