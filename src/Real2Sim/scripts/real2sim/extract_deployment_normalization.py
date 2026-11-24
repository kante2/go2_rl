#!/usr/bin/env python3
"""Extract fixed ONNX input Sub/Div constants; no inference or simulation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
from onnx.numpy_helper import to_array


def extract(directory):
    path=directory/'policy.onnx'
    meta=json.loads((directory/'policy.json').read_text())
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != meta['onnx_sha256']:
        raise ValueError('ONNX hash mismatch')
    model=onnx.load(path)
    first,second=model.graph.node[:2]
    if first.op_type!='Sub' or second.op_type!='Div' or second.input[0]!=first.output[0] or first.input[0]!=model.graph.input[0].name:
        raise ValueError('Unsupported input normalization graph')
    weights={v.name:to_array(v) for v in model.graph.initializer}
    mean=weights[first.input[1]].reshape(-1); denominator=weights[second.input[1]].reshape(-1)
    if mean.shape!=(47,) or denominator.shape!=(47,) or not np.isfinite(mean).all() or not np.isfinite(denominator).all() or np.any(denominator<=0):
        raise ValueError('Invalid normalization constants')
    return dict(onnx_sha256=sha,checkpoint_sha256=meta['checkpoint_sha256'],mean=mean.tolist(),denominator=denominator.tolist(),
        source='Actual ONNX input Sub/Div initializers; denominator already includes epsilon',
        mean_initializer=first.input[1],denominator_initializer=second.input[1],inference_performed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy-dir',type=Path,required=True)
    print(json.dumps(extract(p.parse_args().policy_dir),indent=2))
