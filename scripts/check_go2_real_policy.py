#!/usr/bin/env python3
"""Verify ONNX inference and SDK observation mapping against exported MuJoCo frames."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from go2_autonomous_driving.real.real_control import PolicyMapping
from go2_autonomous_driving.real.real_quietpaw import QuietPawMapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'artifacts/go2_real')
    directory = parser.parse_args().policy_dir
    metadata = json.loads((directory / 'policy.json').read_text())
    quietpaw = metadata.get('policy_family') == 'quietpaw'
    mapping = QuietPawMapping(metadata) if quietpaw else PolicyMapping(metadata)
    if hashlib.sha256((directory / 'policy.onnx').read_bytes()).hexdigest() != metadata['onnx_sha256']:
        raise SystemExit('ONNX hash does not match metadata')
    data = np.load(directory / 'parity.npz')
    samples = json.loads((directory / 'samples.json').read_text())
    if (not samples or data['observations'].shape != (len(samples), mapping.observation_size)
            or data['actions'].shape != (len(samples), 12)):
        raise SystemExit('Invalid or empty policy parity samples')
    session = ort.InferenceSession(str(directory / 'policy.onnx'), providers=['CPUExecutionProvider'])
    key = session.get_inputs()[0].name
    output = np.concatenate([session.run(None, {key: obs[None, :]})[0] for obs in data['observations']])
    policy_error = float(np.max(np.abs(output - data['actions'])))
    mapping_error = 0.0
    for sample, obs in zip(samples, data['observations']):
        if quietpaw:
            mapping = QuietPawMapping(metadata, sample['quietness'])
        actual = mapping.observation(mapping.to_sdk(sample['q']), mapping.to_sdk(sample['dq']),
                                     sample['quaternion'], sample['gyro'], obs[6:9],
                                     sample['step'], obs[35:47])[0]
        mapping_error = max(mapping_error, float(np.max(np.abs(actual - obs))))
    print(json.dumps(dict(frames=len(samples), onnx_torch_max_error=policy_error,
                          real_observation_max_error=mapping_error), indent=2))
    if not np.isfinite(output).all() or not np.isfinite(data['actions']).all():
        raise SystemExit('Non-finite policy parity output')
    if not np.isfinite(data['observations']).all() or policy_error >= 1e-4 or mapping_error >= 1e-4:
        raise SystemExit('Policy parity failed')


if __name__ == '__main__':
    main()
