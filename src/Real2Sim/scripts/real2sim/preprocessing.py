"""Load, filter and sample recorded logs without inference or simulation."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation, Slerp
from .config import PROJECT, DEFAULT_DATASET, JOINTS, LEGS

SDK_ORDER = 'SDK: FR, FL, RR, RL; hip, thigh, calf per leg'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_raw(path):
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files if k != 'meta'}
        meta = json.loads(z['meta'].item())
    return data, meta

def event(data, label):
    return float(data['ev_t'][np.flatnonzero(data['ev_label'] == label)[0]])

def interp(t, values, query):
    return np.stack([np.interp(query, t, values[:, j]) for j in range(values.shape[1])], axis=1)

def filtered(t, values, dt, cutoff):
    uniform = np.arange(t[0], t[-1] + 1e-9, dt)
    signal = interp(t, values, uniform)
    return uniform, sosfiltfilt(butter(4, cutoff, fs=1 / dt, output='sos'), signal, axis=0)

def prepare_legacy_pair(real, sim, cutoff, dt):
    origin = event(real, 'record_start')
    tick = real['ls_tick'].astype(float)
    if np.any(np.diff(tick) < 0):
        raise ValueError('Tick reset/wrap; split recording first')
    keep = np.r_[np.diff(tick) != 0, True]
    clock = (tick[keep] - tick[0]) * .001
    offset = np.median(real['ls_t'][keep] - clock)
    tr = clock + offset - origin
    ts = sim['ls_t']
    # Different sample rates: filter each native-rate uniform signal first.
    qr_t, qr = filtered(tr, real['ls_q'][keep], .002, cutoff)
    dr_t, dr = filtered(tr, real['ls_dq'][keep], .002, cutoff)
    qs_t, qs = filtered(ts, sim['ls_q'], dt, cutoff)
    ds_t, ds = filtered(ts, sim['ls_dq'], dt, cutoff)
    ar = np.gradient(dr, .002, axis=0)
    ass = np.gradient(ds, dt, axis=0)
    start = max(qr_t[0], qs_t[0])
    end = min(qr_t[-1], qs_t[-1])
    t = np.arange(np.ceil(start / dt) * dt, end, dt)
    arrays = dict(t=t, real_q=interp(qr_t, qr, t), sim_q=interp(qs_t, qs, t),
                  real_dq=interp(dr_t, dr, t), sim_dq=interp(ds_t, ds, t),
                  real_ddq=interp(dr_t, ar, t), sim_ddq=interp(ds_t, ass, t),
                  real_tau=interp(*filtered(tr, real['ls_tau'][keep], .002, cutoff), t),
                  sim_tau=interp(*filtered(ts, sim['ls_tau'], dt, cutoff), t),
                  real_foot_sensor_raw=interp(tr, real['ls_ff'][keep], t),
                  sim_foot_fz_N=interp(ts, sim['sim_foot_fz_N'], t))
    arrays['real_tau_filtered'] = arrays['real_tau']  # Existing force-estimator alias.
    arrays['real_quaternion'] = Slerp(tr, Rotation.from_quat(
        real['ls_quaternion'][keep][:, [1, 2, 3, 0]]))(t).as_quat()[:, [3, 0, 1, 2]]
    valid = (t > start + .2) & (t < end - .2)
    gaps = np.flatnonzero(np.diff(tr) > .010)
    for index in gaps:
        # Also exclude filter neighborhoods around a missing-data interval.
        valid &= ~((t >= tr[index] - .2) & (t <= tr[index + 1] + .2))
    move_start, move_stop = event(real, 'move_start') - origin, event(real, 'move_stop') - origin
    arrays['valid'] = valid
    arrays['moving'] = valid & (t >= move_start) & (t <= move_stop)
    arrays['steady'] = valid & (t >= move_start + 2) & (t <= move_stop - 1)
    return arrays, dict(move_start_s=move_start, move_stop_s=move_stop,
                        duplicate_ticks_removed=int((~keep).sum()), large_gaps=int(len(gaps)),
                        max_tick_gap_ms=float(np.diff(tr).max() * 1000),
                        tick_to_receive_offset_s=float(offset),
                        receive_minus_tick_median_ms=float(np.median(real['ls_t'][keep] - clock - offset) * 1000))

def adapt(data, meta):
    if meta.get('controller') == 'rl' and 'timestamp_clock' in meta:
        meta = dict(meta, joint_order=SDK_ORDER, shared_pc_clock=True)
    elif meta.get('source') == 'mujoco' and 'sim_time' in data:
        # S3 records the opposite contact-vector direction: loaded feet have
        # negative z. Convert the known recording convention to positive load.
        # Do not use abs(), which would conceal an unexpected input convention.
        recorded_z = data['sim_foot_force_vec'][:, :, 2]
        if not np.isfinite(recorded_z).all() or np.any(recorded_z > 1e-6):
            raise ValueError('Unexpected S3 contact-force convention: recorded z must be non-positive')
        load_z = -recorded_z
        load_z[load_z == 0] = 0.  # Canonical positive zero, without changing raw arrays.
        data = dict(data, sim_foot_fz_N=load_z)
        meta = dict(meta, joint_order=JOINTS, foot_order=['FR', 'FL', 'RR', 'RL'],
                    shared_pc_clock=True,
                    force_source='sim_foot_force_vec[:, :, 2]', force_sign_to_load=-1.,
                    force_semantics='S3 recorded negative-z contact vector negated to positive vertical load [N]; no airborne offset')
    return data, meta

def prepare_recorded_pair(real, sim, cutoff, dt):
    # move_start/stop delimit the recording, not a commanded walking interval.
    origin = event(real, 'move_start')
    tick = real['ls_tick'].astype(float)
    if np.any(np.diff(tick) < 0):
        raise ValueError('Tick reset/wrap; split recording first')
    keep = np.r_[np.diff(tick) != 0, True]
    clock = (tick[keep] - tick[0]) * .001
    offset = np.median(real['ls_t'][keep] - clock)
    tr = clock + offset - origin
    ts = sim['ls_t'] - origin
    physics = sim['sim_time']
    if np.any(np.diff(ts) <= 0) or np.any(np.diff(physics) <= 0):
        raise ValueError('Non-monotonic simulation clocks')
    start, end = max(tr[0], ts[0]), min(tr[-1], ts[-1])
    if end - start < 1.:
        raise ValueError('Less than one second of shared-clock overlap')
    t = np.arange(np.ceil(start / dt) * dt, end, dt)
    arrays = dict(t=t)
    derivatives = {}
    for domain, raw, native, receive, step in (
            ('real', real, tr, tr, .002), ('sim', sim, physics, ts, dt)):
        for quantity in ('q', 'dq', 'tau'):
            values = raw['ls_' + quantity][keep] if domain == 'real' else raw['ls_' + quantity]
            uniform, signal = filtered(native, values, step, cutoff)
            # Filter/differentiate in physical time, then use PC receive timestamps
            # to compare the recordings. Do not treat batched receive dt as physics dt.
            sampled = interp(uniform, signal, native)
            arrays[domain + '_' + quantity] = interp(receive, sampled, t)
            if quantity == 'dq':
                acceleration = interp(uniform, np.gradient(signal, step, axis=0), native)
                derivatives[domain] = interp(receive, acceleration, t)
        arrays[domain + '_ddq'] = derivatives[domain]
    arrays['real_foot_sensor_raw'] = interp(tr, real['ls_ff'][keep], t)
    arrays['sim_foot_fz_N'] = interp(ts, sim['sim_foot_fz_N'], t)
    valid = (t > start + .2) & (t < end - .2)
    gaps = np.flatnonzero(np.diff(tr) > .010)
    for index in gaps:
        valid &= ~((t >= tr[index] - .2) & (t <= tr[index + 1] + .2))
    sim_gaps = np.flatnonzero((np.diff(physics) > dt * 1.5) | (np.diff(ts) > .050))
    for index in sim_gaps:
        valid &= ~((t >= ts[index] - .2) & (t <= ts[index + 1] + .2))
    move_start, move_stop = 0., event(real, 'move_stop') - origin
    arrays['valid'] = valid
    arrays['moving'] = valid & (t >= move_start) & (t <= move_stop)
    arrays['steady'] = valid & (t >= move_start + 2.) & (t <= move_stop - 1.)
    return arrays, dict(move_start_s=move_start, move_stop_s=move_stop,
        duplicate_ticks_removed=int((~keep).sum()), large_gaps=int(len(gaps)),
        sim_large_gaps=int(len(sim_gaps)), shared_pc_clock=True,
        max_tick_gap_ms=float(np.diff(tr).max() * 1000),
        tick_to_receive_offset_s=float(offset),
        overlap_s=[float(start), float(end)],
        evaluation_scope='recording interior, excluding first 2s / last 1s and data gaps; walking not verified',
        sim_filter_clock='sim_time; filtered signals mapped to shared PC ls_t',
        receive_minus_tick_median_ms=float(np.median(real['ls_t'][keep] - clock - offset) * 1000))

def load(path):
    return adapt(*load_raw(path))

def prepare_pair(real, sim, cutoff, dt):
    if 'sim_time' in sim:
        return prepare_recorded_pair(real, sim, cutoff, dt)
    return prepare_legacy_pair(real, sim, cutoff, dt)

def named_mapping(real, rm, sim, sm):
    """Physical-unit logs, not actor actions; never apply action_scale to q."""
    expected = 'SDK: FR, FL, RR, RL; hip, thigh, calf per leg'
    if rm.get('joint_order') != expected:
        raise ValueError('Unrecognized real joint ordering')
    order = sm.get('joint_order', [])
    feet = sm.get('foot_order', [])
    if len(order) != 12 or set(order) != set(JOINTS) or len(feet) != 4 or set(feet) != set(LEGS):
        raise ValueError('Unrecognized sim joint/foot ordering')
    ji, fi = [order.index(j) for j in JOINTS], [feet.index(f) for f in LEGS]
    mapped = dict(sim)
    for key in ('ls_q', 'ls_dq', 'ls_tau'):
        mapped[key] = sim[key][:, ji]
    mapped['sim_foot_fz_N'] = sim['sim_foot_fz_N'][:, fi]
    contract = dict(joint_indices=ji, foot_indices=fi, joint_order=list(JOINTS),
        foot_order=list(LEGS), q_unit='rad', dq_unit='rad/s', ddq_unit='rad/s²',
        joint_sign='SDK/MJCF corresponding joint axes; no fitted sign or zero offset',
        joint_gain=1., joint_offset_rad=0., action_scale_applied=False,
        real_force='raw foot sensor counts; scalar sensor direction not reconstructed',
        sim_force=sm['force_semantics'], force_calibration_applied=False,
        sim_force_source=sm.get('force_source', 'sim_foot_fz_N'),
        sim_force_sign_to_load=sm.get('force_sign_to_load', 1.),
        filter='both q/dq: fourth-order zero-phase 15 Hz LPF; ddq: derivative of filtered dq',
        resampling_dt_s=.005, simulation_executed=False)
    return real, mapped, contract

def paired_paths(data_dir, dataset=DEFAULT_DATASET):
    """Discover all pairs inside exactly one dataset, never another session."""
    if not dataset or Path(dataset).name != dataset or dataset in ('.', '..'):
        raise ValueError('dataset must be a single folder name')
    pairs = []
    for real in sorted((data_dir / 'real' / dataset).glob('*.npz')):
        sim_name = real.name[:-len('_real.npz')] + '_sim.npz' if real.name.endswith('_real.npz') else real.name
        pairs.append((real, data_dir / 'sim' / dataset / sim_name))
    if not pairs:
        raise FileNotFoundError(f'No real NPZ recordings in {data_dir / "real" / dataset}')
    missing = [str(path) for pair in pairs for path in pair if not path.is_file()]
    if missing:
        raise FileNotFoundError('Required real/sim trial files are missing:\n' + '\n'.join(missing))
    return pairs

