#!/usr/bin/env python3
"""Verify ONNX inference and SDK observation mapping against exported MuJoCo frames."""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from go2_autonomous_driving.real.real_control import PolicyMapping


def main():
    directory = Path(__file__).resolve().parents[1] / 'artifacts/go2_real'
    mapping = PolicyMapping(json.loads((directory / 'policy.json').read_text()))
    data = np.load(directory / 'parity.npz')
    samples = json.loads((directory / 'samples.json').read_text())
    session = ort.InferenceSession(str(directory / 'policy.onnx'), providers=['CPUExecutionProvider'])
    key = session.get_inputs()[0].name
    output = np.concatenate([session.run(None, {key: obs[None, :]})[0] for obs in data['observations']])
    policy_error = float(np.max(np.abs(output - data['actions'])))
    mapping_error = 0.0
    for sample, obs in zip(samples, data['observations']):
        actual = mapping.observation(mapping.to_sdk(sample['q']), mapping.to_sdk(sample['dq']),
                                     sample['quaternion'], sample['gyro'], obs[6:9],
                                     sample['step'], obs[35:47])[0]
        mapping_error = max(mapping_error, float(np.max(np.abs(actual - obs))))
    print(json.dumps(dict(frames=len(samples), onnx_torch_max_error=policy_error,
                          real_observation_max_error=mapping_error), indent=2))
    if policy_error >= 1e-4 or mapping_error >= 1e-4:
        raise SystemExit('Policy parity failed')


if __name__ == '__main__':
    main()
