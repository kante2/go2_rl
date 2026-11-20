"""File-only deployment coordinate checks; never infer a policy or step physics."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SDK_JOINTS = tuple(f'{leg}_{joint}_joint' for leg in ('FR','FL','RR','RL') for joint in ('hip','thigh','calf'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_contract(checkpoint, directory=None, normalization_path=None):
    directory = Path(directory) if directory else ROOT/'artifacts/go2_real/baseline'/checkpoint
    meta = json.loads((directory/'policy.json').read_text())
    order = np.asarray(meta['policy_to_sdk'],dtype=int)
    default = np.asarray(meta['default_joint_pos'],dtype=float)
    scale = np.broadcast_to(np.asarray(meta['action_scale'],dtype=float),(12,)).copy()
    if meta['checkpoint_sha256'] != checkpoint or digest(directory/'policy.onnx') != meta['onnx_sha256']:
        raise ValueError('Deployment artifact hash mismatch')
    if sorted(order.tolist()) != list(range(12)) or default.shape != (12,) or not np.isfinite(default).all() or not np.isfinite(scale).all() or np.any(scale == 0):
        raise ValueError('Invalid deployment mapping')
    if [SDK_JOINTS[i] for i in order] != meta['joint_names']:
        raise ValueError('Deployment joint names disagree with policy_to_sdk')
    if meta['observation_names'] != ['base_ang_vel','projected_gravity','command','phase','joint_pos','joint_vel','actions']:
        raise ValueError('Unknown observation layout')
    samples = json.loads((directory/'samples.json').read_text())
    with np.load(directory/'parity.npz',allow_pickle=False) as z:
        obs = z['observations']; actions = z['actions']
    if obs.shape != (len(samples),47) or actions.shape != (len(samples),12) or not len(samples):
        raise ValueError('Invalid deployment parity samples')
    q = np.asarray([s['q'] for s in samples],dtype=np.float32)
    dq = np.asarray([s['dq'] for s in samples],dtype=np.float32)
    error = max(float(abs(q-default.astype(np.float32)-obs[:,11:23]).max()),float(abs(dq-obs[:,23:35]).max()))
    if error > 1e-6:
        raise ValueError('Deployment joint observation parity failed')
    normalization_path = Path(normalization_path) if normalization_path else Path(__file__).with_name('deployment_normalization.json')
    norm = json.loads(normalization_path.read_text())
    if norm['onnx_sha256'] != meta['onnx_sha256'] or norm['checkpoint_sha256'] != checkpoint:
        raise ValueError('Normalization constants do not belong to this deployed ONNX')
    mean = np.asarray(norm['mean'],dtype=float)
    denominator = np.asarray(norm['denominator'],dtype=float)
    if mean.shape != (47,) or denominator.shape != (47,) or not np.isfinite(mean).all() or not np.isfinite(denominator).all() or np.any(denominator <= 0):
        raise ValueError('Invalid normalization constants')
    provenance = dict(policy_directory=str(directory.relative_to(ROOT)) if directory.is_relative_to(ROOT) else str(directory),
        checkpoint_sha256=checkpoint,
        assets={name:digest(directory/name) for name in ('policy.json','policy.onnx','samples.json','parity.npz')},
        policy_to_sdk=order.tolist(),policy_joint_names=meta['joint_names'],
        default_joint_pos=default.tolist(),action_scale=scale.tolist(),
        observation_formula='q_policy = q_sdk[policy_to_sdk] - default; dq_policy = dq_sdk[policy_to_sdk]',
        action_formula='target_policy = default + action_scale * action; target_sdk[policy_to_sdk] = target_policy',
        plotted_space='q/dq: normalized actor input; ddq: physical rad/s² (no deployment mapping)',
        normalization_mean=mean.tolist(), normalization_denominator=denominator.tolist(),
        normalization_sha256=digest(normalization_path),
        normalization_source=norm['source'],
        action_scale_applied_to_measured_q=False,
        ddq='not a policy input; joint permutation only; no gain or offset',
        tau='ls_tau in N·m; not a policy input; joint permutation only; no deployment gain or offset',
        foot_force='not a policy input; no deployment counts-to-N mapping; unchanged',
        normalization='Apply actual ONNX input Sub/Div once to q/dq of both domains; no inverse numeric transform in plots',
        joint_parity_frames=len(samples),joint_observation_parity_max_error=error,
        policy_inference_performed=False, fitted_parameters=False)
    return dict(order=order,default=default,scale=scale,mean=mean,denominator=denominator,provenance=provenance)


def to_sdk(values,order):
    result = np.empty_like(values)
    result[...,order] = values
    return result


def apply_contract(data, contract, real, sim):
    """Expose actual normalized actor inputs. Only permute names for UI; do not invert numeric mapping."""
    output = {k:v.copy() for k,v in data.items()}
    order,default,scale = (contract[k] for k in ('order','default','scale'))
    coordinates = {}; errors = {}
    for domain, raw in (('real',real),('sim',sim)):
        q_policy = data[domain+'_q'][:,order]-default
        dq_policy = data[domain+'_dq'][:,order]
        q_normalized = (q_policy-contract['mean'][11:23])/contract['denominator'][11:23]
        dq_normalized = (dq_policy-contract['mean'][23:35])/contract['denominator'][23:35]
        # to_sdk only labels joint names; normalization/offset are NOT undone.
        output[domain+'_q'] = to_sdk(q_normalized,order)
        output[domain+'_dq'] = to_sdk(dq_normalized,order)
        output[domain+'_ddq'] = to_sdk(data[domain+'_ddq'][:,order],order)
        output[domain+'_tau'] = to_sdk(data[domain+'_tau'][:,order],order)
        coordinates[domain+'_q_policy_relative'] = q_policy
        coordinates[domain+'_dq_policy'] = dq_policy
        coordinates[domain+'_q_policy_normalized'] = q_normalized
        coordinates[domain+'_dq_policy_normalized'] = dq_normalized
        errors[domain+'_q_roundtrip_max_error_rad'] = float(abs(to_sdk(q_policy+default,order)-data[domain+'_q']).max())
        errors[domain+'_dq_roundtrip_max_error_rad_s'] = float(abs(to_sdk(dq_policy,order)-data[domain+'_dq']).max())
        # Target coordinates use action_scale; these are not historical raw policy actions.
        target = raw['lc_q'][:,order]
        action_coordinate = (target-default)/scale
        reconstructed = to_sdk(default+scale*action_coordinate,order)
        errors[domain+'_target_roundtrip_max_error_rad'] = float(abs(reconstructed-raw['lc_q']).max())
        coordinates[domain+'_logged_target_action_coordinate'] = action_coordinate
        coordinates[domain+'_logged_target_t'] = raw['lc_t']
    if max(errors.values()) > 1e-10:
        raise ValueError('Deployment round trip failed')
    return output,coordinates,errors
