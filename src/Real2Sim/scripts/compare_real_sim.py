#!/usr/bin/env python3
"""Compare paired physical-unit logs, with explicitly approximate force estimates.

Run in the Go2 container (NumPy, SciPy, Matplotlib and CPU MuJoCo).
No real-robot connection, policy execution or simulator parameter fitting.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation, Slerp
from phase_synchronization import estimate_alignment, apply_alignment, plot_alignment

ROOT = Path(__file__).resolve().parents[3]
LEGS = ('FR', 'FL', 'RR', 'RL')
JOINTS = tuple(f'{leg}_{j}_joint' for leg in LEGS for j in ('hip', 'thigh', 'calf'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
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


def correlation(a, b):
    if len(a) < 3 or np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def statistics(a, b):
    delta = b - a
    return dict(rmse=float(np.sqrt(np.mean(delta ** 2))), mae=float(np.mean(abs(delta))),
                bias_sim_minus_real=float(np.mean(delta)), pearson=correlation(a, b),
                real_mean=float(np.mean(a)), sim_mean=float(np.mean(b)),
                real_std=float(np.std(a)), sim_std=float(np.std(b)),
                real_p95_minus_p05=float(np.percentile(a, 95) - np.percentile(a, 5)),
                sim_p95_minus_p05=float(np.percentile(b, 95) - np.percentile(b, 5)))


def force_solve(jacobian, residual, damping=.005):
    """DLS solution of J.T F = gravity - estimated motor torque; no clipping."""
    return np.linalg.solve(jacobian @ jacobian.T + damping ** 2 * np.eye(3),
                           jacobian @ residual)


class QuasiStaticForce:
    def __init__(self, model_path, damping):
        import mujoco
        self.mj = mujoco
        self.damping = damping
        # Use the same nominal MJCF geometry as the simulation collector.
        root = ET.parse(model_path).getroot()
        compiler = root.find('compiler')
        compiler.set('meshdir', str((model_path.parent / compiler.get('meshdir', '.')).resolve()))
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.data = mujoco.MjData(self.model)
        self.qids = [self.model.jnt_qposadr[self.model.joint(j).id] for j in JOINTS]
        self.dids = [self.model.jnt_dofadr[self.model.joint(j).id] for j in JOINTS]
        self.sites = [self.model.site(leg).id for leg in LEGS]

    def estimate(self, q, quaternion, tau):
        mj, m, d = self.mj, self.model, self.data
        naive, compensated, condition, residual = [], [], [], []
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for angles, quat, torque in zip(q, quaternion, tau):
            d.qpos[:3] = [0, 0, 1]
            d.qpos[3:7] = quat / np.linalg.norm(quat)
            d.qpos[self.qids] = angles
            d.qvel[:] = 0  # Explicit quasi-static approximation, not inverse dynamics.
            mj.mj_fwdPosition(m, d)
            mj.mj_fwdVelocity(m, d)
            gravity = d.qfrc_bias[self.dids].copy()
            f0, fg, cond, res = [], [], [], []
            for leg, sid in enumerate(self.sites):
                mj.mj_jacSite(m, d, jp, jr, sid)
                sl = slice(3 * leg, 3 * leg + 3)
                jac = jp[:, np.asarray(self.dids)[sl]]
                target = gravity[sl] - torque[sl]
                estimate = force_solve(jac, target, self.damping)
                f0.append(force_solve(jac, -torque[sl], self.damping))
                fg.append(estimate)
                cond.append(np.linalg.cond(jac))
                res.append(np.linalg.norm(jac.T @ estimate - target))
            naive.append(f0); compensated.append(fg); condition.append(cond); residual.append(res)
        return dict(force_tau_only_world_N=np.asarray(naive),
                    force_gravity_comp_world_N=np.asarray(compensated),
                    jacobian_condition=np.asarray(condition),
                    joint_torque_residual_Nm=np.asarray(residual))


def prepare_pair(real, sim, cutoff, dt):
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


def save_figure(fig, path):
    import matplotlib.pyplot as plt
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def joint_figure(a, trial, quantity, output, window=None):
    import matplotlib.pyplot as plt
    units = dict(q='rad', dq='rad/s', ddq='rad/s²')
    t = a['t']
    mask = np.ones(len(t), bool) if window is None else (t >= window[0]) & (t <= window[1])
    fig, axes = plt.subplots(4, 3, figsize=(16, 11), sharex=True)
    for j, ax in enumerate(axes.flat):
        ax.plot(t[mask], a['real_' + quantity][mask, j], label='Real', lw=.8)
        ax.plot(t[mask], a['sim_' + quantity][mask, j], label='Simulation', lw=.8, alpha=.8)
        ax.set_title(JOINTS[j].replace('_joint', ''))
        ax.set_ylabel(units[quantity]); ax.grid(alpha=.25)
    for ax in axes[-1]: ax.set_xlabel('Time from record start [s]')
    axes[0, 0].legend()
    fig.suptitle(f'{trial}: {quantity} | 15 Hz low-pass; ddq derived from dq | common footfall phase shift')
    fig.tight_layout()
    save_figure(fig, output)


def zscore(a, mask):
    std = np.std(a[mask], axis=0)
    return (a - np.mean(a[mask], axis=0)) / np.where(std > 1e-10, std, 1.)


def force_figure(a, trial, kind, output, window=None):
    import matplotlib.pyplot as plt
    t, steady = a['t'], a['steady']
    mask = np.ones(len(t), bool) if window is None else (t >= window[0]) & (t <= window[1])
    sensor = a['real_foot_sensor_raw']
    other = a['force_gravity_comp_world_N'][:, :, 2] if kind == 'estimated' else a['sim_foot_fz_N']
    label = 'Real quasi-static estimate' if kind == 'estimated' else 'Simulation GRF'
    fig, axes = plt.subplots(4, 2, figsize=(16, 12), sharex=True)
    zn, zo = zscore(sensor, steady), zscore(other, steady)
    for foot, leg in enumerate(LEGS):
        ax = axes[foot, 0]
        line1 = ax.plot(t[mask], sensor[mask, foot], color='C0', lw=.8, label='Real foot sensor [counts]')
        ax.set_ylabel(f'{leg}: sensor [counts]', color='C0')
        right = ax.twinx()
        line2 = right.plot(t[mask], other[mask, foot], color='C1', lw=.8, alpha=.8, label=label + ' [N]')
        right.set_ylabel('World Fz [N]', color='C1')
        ax.grid(alpha=.2)
        if foot == 0: ax.legend(line1 + line2, [l.get_label() for l in line1 + line2], fontsize=7)
        ax = axes[foot, 1]
        ax.plot(t[mask], zn[mask, foot], label='Real sensor', lw=.8)
        ax.plot(t[mask], zo[mask, foot], label=label, lw=.8, alpha=.8)
        ax.set_ylabel(f'{leg}: z-score'); ax.grid(alpha=.2)
    for ax in axes[-1]: ax.set_xlabel('Time from record start [s]')
    axes[0, 1].legend(fontsize=7)
    fig.suptitle(f'{trial}: real sensor vs {label}\nDifferent units: independent axes; z-score is shape comparison, NOT force calibration')
    fig.tight_layout()
    save_figure(fig, output)


def write_csv(path, rows):
    if not rows: return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def summary_figures(joints, forces, directory):
    import matplotlib.pyplot as plt
    rows = [r for r in joints if r['segment'] == 'steady']
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    for ax, quantity, unit in zip(axes, ('q', 'dq', 'ddq'), ('rad', 'rad/s', 'rad/s²')):
        values = np.array([[r['rmse'] for r in rows if r['quantity'] == quantity and r['joint'] == j] for j in JOINTS])
        ax.bar(np.arange(12), values.mean(axis=1), yerr=values.std(axis=1), capsize=3)
        ax.set_xticks(np.arange(12), [j.replace('_joint', '') for j in JOINTS], rotation=25)
        ax.set_ylabel(f'{quantity} RMSE [{unit}]'); ax.grid(axis='y', alpha=.2)
    fig.suptitle('Steady walking: phase-aligned RMSE, mean ± SD across five paired trials\nOne common start-footfall shift per trial; 15 Hz low-pass; inspect synchronization status')
    fig.tight_layout(); save_figure(fig, directory / 'summary_joint_rmse.png')
    trials = sorted(set(r['trial'] for r in forces))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, kind in zip(axes, ('sensor_vs_quasistatic', 'sensor_vs_sim')):
        values = np.array([[next(r['pearson'] for r in forces if r['trial'] == trial and r['foot'] == leg and r['comparison'] == kind)
                            for leg in LEGS] for trial in trials], dtype=float)
        im = ax.imshow(values, vmin=-1, vmax=1, cmap='coolwarm', aspect='auto')
        ax.set_xticks(range(4), LEGS); ax.set_yticks(range(len(trials)), trials)
        ax.set_title(kind)
        for (i, j), value in np.ndenumerate(values): ax.text(j, i, f'{value:.2f}', ha='center', va='center')
        fig.colorbar(im, ax=ax, label='Pearson r (phase-aligned steady segment)')
    fig.suptitle('Foot force waveform correlation; sensor counts are NOT newtons')
    fig.tight_layout(); save_figure(fig, directory / 'summary_force_correlation.png')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=ROOT / 'src/Real2Sim/data')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'src/Real2Sim/result')
    p.add_argument('--cutoff', type=float, default=15.)
    args = p.parse_args()
    if not 0 < args.cutoff < 100: p.error('cutoff must be between 0 and 100 Hz')
    out, figures = args.output_dir, args.output_dir / 'figure'
    # Preserve original results once; re-runs must not replace the reference.
    original = out / 'unaligned'
    if (out / 'summary.json').exists() and not original.exists():
        existing = json.loads((out / 'summary.json').read_text())
        if existing.get('phase_alignment', '').startswith('record-start'):
            original.mkdir(parents=True)
            for name in ('figure', 'processed'):
                shutil.copytree(out / name, original / name)
            for name in ('summary.json', 'report.md', 'joint_metrics.csv', 'force_metrics.csv'):
                shutil.copy2(out / name, original / name)
    figures.mkdir(parents=True, exist_ok=True)
    sync_dir = out / 'synchronization'; sync_dir.mkdir(exist_ok=True)
    processed = out / 'processed'; processed.mkdir(exist_ok=True)
    model_path = ROOT / 'src/assets/robots/unitree_go2/xmls/go2.xml'
    estimator = QuasiStaticForce(model_path, .005)
    joint_rows, force_rows, inputs = [], [], []
    pairs = sorted((args.data_dir / 'real').glob('*.npz'))
    if not pairs: raise ValueError('No real recordings')
    for number, real_path in enumerate(pairs):
        sim_path = args.data_dir / 'sim' / real_path.name
        real, rm = load(real_path); sim, sm = load(sim_path)
        if not sm['completed'] or rm['checkpoint_sha256'] != sm['checkpoint_sha256']:
            raise ValueError('Incomplete rollout or mismatched policy')
        if sm['reference_sha256'] != digest(real_path): raise ValueError('Real file differs from simulation reference')
        if tuple(sm['joint_order']) != JOINTS or tuple(sm['foot_order']) != LEGS:
            raise ValueError('Unexpected simulation joint/foot order')
        a, info = prepare_pair(real, sim, args.cutoff, .005)
        a.update(estimator.estimate(a['real_q'], a['real_quaternion'], a['real_tau_filtered']))
        trial = real_path.stem
        synchronization, contact_events = estimate_alignment(a, info['move_start_s'], info['move_stop_s'])
        before_alignment = a
        a = apply_alignment(a, synchronization['applied_lag_s'])
        (sync_dir / f'{trial}.json').write_text(json.dumps(synchronization, indent=2, allow_nan=False) + '\n')
        write_csv(sync_dir / f'{trial}_events.csv', contact_events)
        plot_alignment(before_alignment, a, synchronization, contact_events,
                       figures / f'{trial}_synchronization.png')
        info['synchronization'] = synchronization
        for segment in ('moving', 'steady'):
            mask = a[segment]
            for quantity, unit in (('q', 'rad'), ('dq', 'rad/s'), ('ddq', 'rad/s^2')):
                for j, joint in enumerate(JOINTS):
                    joint_rows.append(dict(trial=trial, segment=segment, quantity=quantity, unit=unit,
                        joint=joint, samples=int(mask.sum()),
                        **statistics(a['real_' + quantity][mask, j], a['sim_' + quantity][mask, j])))
        mask = a['steady']
        for foot, leg in enumerate(LEGS):
            for kind, estimate in (('sensor_vs_quasistatic', a['force_gravity_comp_world_N'][:, :, 2]),
                                   ('sensor_vs_sim', a['sim_foot_fz_N'])):
                sensor, force = a['real_foot_sensor_raw'][mask, foot], estimate[mask, foot]
                force_rows.append(dict(trial=trial, comparison=kind, foot=leg, samples=int(mask.sum()),
                    sensor_unit='raw_counts', force_unit='N', absolute_force_rmse_N=None,
                    pearson=correlation(sensor, force), sensor_mean_counts=float(np.mean(sensor)),
                    force_mean_N=float(np.mean(force)), force_p95_N=float(np.percentile(force, 95)),
                    negative_force_fraction=float(np.mean(force < 0)),
                    jacobian_condition_median=float(np.median(a['jacobian_condition'][mask, foot]))))
        inputs.append(dict(trial=trial, real_sha256=digest(real_path), sim_sha256=digest(sim_path), **info))
        np.savez_compressed(processed / f'{trial}_comparison.npz', **a)
        for quantity in ('q', 'dq', 'ddq'):
            joint_figure(a, trial, quantity, figures / f'{trial}_{quantity}.png')
        for kind in ('estimated', 'sim'):
            force_figure(a, trial, kind, figures / f'{trial}_sensor_vs_{kind}.png')
        if number == 0:
            zoom = (info['move_start_s'] + 3, info['move_start_s'] + 6)
            for quantity in ('q', 'dq', 'ddq'):
                joint_figure(a, trial, quantity, figures / f'{trial}_{quantity}_zoom.png', zoom)
            for kind in ('estimated', 'sim'):
                force_figure(a, trial, kind, figures / f'{trial}_sensor_vs_{kind}_zoom.png', zoom)
        print(f'{trial}: {synchronization["applied_lag_s"] * 1000:+.1f} ms, '
              f'{synchronization["status"]}; compared {len(a["t"])} samples', flush=True)
    write_csv(out / 'joint_metrics.csv', joint_rows)
    write_csv(out / 'force_metrics.csv', force_rows)
    summary_figures(joint_rows, force_rows, figures)
    report(out, inputs, joint_rows, force_rows, model_path, args.cutoff)
    print(f'Results: {out / "report.md"}', flush=True)


def report(out, inputs, joints, forces, model_path, cutoff):
    aggregate = {}
    for quantity in ('q', 'dq', 'ddq'):
        rows = [r for r in joints if r['quantity'] == quantity and r['segment'] == 'steady']
        aggregate[quantity] = float(np.sqrt(sum(r['rmse'] ** 2 * r['samples'] for r in rows) /
                                           sum(r['samples'] for r in rows)))
    summary = dict(schema='go2-real-sim-comparison-v1', trials=inputs,
                   joint_steady_pooled_rmse=aggregate, common_rate_hz=200, cutoff_hz=cutoff,
                   derivative='ddq = time derivative of low-pass filtered dq; not measured acceleration',
                   force_method='Quasi-static J.T F = g(q) - tau_est; world frame; DLS lambda=0.005 m',
                   model=str(model_path), model_sha256=digest(model_path), force_sensor_calibrated=False,
                   phase_alignment='One constant same-foot-event phase shift per trial; start fit, end held out; not hardware clock sync')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    lines = ['# 실제 Go2와 시뮬레이션 비교', '',
             'baseline model_34997.pt, 입력 0.2 m/s, 실제/시뮬레이션 기록 5쌍을 비교했다.', '',
             '## 관절 관측 결과', '',
             '| 항목 | 정상 보행 pooled RMSE | 단위 |', '|---|---:|---|']
    for quantity, unit in (('q', 'rad'), ('dq', 'rad/s'), ('ddq', 'rad/s²')):
        lines.append(f'| {quantity} | {aggregate[quantity]:.6f} | {unit} |')
    lines += ['', '위 수치는 시행별 공통 착지 위상차를 보정한 뒤의 sim−real 오차다. '
              'q/dq/ddq에 개별적으로 최적 시간차를 맞추지 않았다. 검증 상태가 REVIEW_REQUIRED인 시행은 잠정 정렬이다.', '',
              '[관절별 RMSE 요약](figure/summary_joint_rmse.png)', '',
              '## 시간·보행 위상 정렬', '',
              '`db_preprocessing/scripts/time_synchronization.py`의 시작 이벤트 추정·끝 구간 검증 방식을 참고하고 '
              '`contact_detection.py`의 hysteresis/지속시간 접촉 검출기를 재사용했다. '
              '이번 데이터에는 오디오가 없어 동일 발의 실제 센서 접촉과 시뮬레이션 접촉을 대응시켰다.', '',
              '- 보정 추정: move_start+2초부터 +7초까지의 5초 구간. 끝 5초(move_stop−6초~−1초)는 미사용 검증.',
              '- 발별 임계값은 시작 구간의 10/90 percentile 범위에서 on=15%, off=7%, 접촉 확인 6ms를 사용했다.',
              '- 200 Hz와 도메인별 접촉 파형 차이를 고려해 합의 반경 25ms, MAD 기준 15ms, 끝 잔차 p90 기준 25ms를 명시했다. '
              '기존 오디오 동기화의 더 엄격한 기준을 그대로 적용한 것은 아니다.',
              '- 검색 범위 ±280ms에서 동일 발 이벤트를 일대일 대응하고 단 하나의 offset을 구했다. '
              '이 값은 하드웨어 지연이 아닌 주기적 보행의 상대 위상이며 주기 모호성이 남는다.',
              '- sim_aligned(t)=sim_original(t+lag). 모든 시뮬레이션 q/dq/ddq/힘에 동일 적용. '
              '실제 센서와 실제 준정적 추정력의 시간 관계는 바꾸지 않았다.',
              '- clock drift/time warp와 진폭 보정은 하지 않았고, 겹치지 않는 경계는 잘라냈다.',
              '- 시작 구간에서 후보는 있으나 정밀도/끝 검증이 부족하면 잠정 offset을 표시하고 REVIEW_REQUIRED로 남겼다. '
              '이는 성공한 센서 시간 동기화라고 주장하는 결과가 아니다.', '',
              '| 시행 | 적용 lag (ms) | 끝 구간 잔차 p90 (ms) | 상태 |', '|---|---:|---:|---|']
    for entry in inputs:
        s = entry['synchronization']
        p90 = s['held_out_abs_residual_p90_s']
        p90_text = f'{p90 * 1000:.1f}' if p90 is not None else 'N/A'
        lines.append(f'| {entry["trial"]} | {s["applied_lag_s"] * 1000:+.1f} | {p90_text} | {s["status"]} |')
    if (out / 'unaligned/summary.json').exists():
        old = json.loads((out / 'unaligned/summary.json').read_text())['joint_steady_pooled_rmse']
        lines += ['', '| 항목 | 정렬 전 RMSE | 정렬 후 RMSE |', '|---|---:|---:|']
        for quantity in ('q', 'dq', 'ddq'):
            lines.append(f'| {quantity} | {old[quantity]:.6f} | {aggregate[quantity]:.6f} |')
        lines += ['', '정렬 전 원본 결과는 [unaligned/](unaligned/report.md)에 보존했다. '
                  '정렬 후 수치에는 calibration과 held-out 구간이 모두 포함되므로 완전한 미사용 평가 점수가 아니다.', '']
    lines += [
              '## 실제 센서와 힘 추정/시뮬레이션 비교', '',
              '**실제 발 센서는 raw counts이며 N 단위 보정 자료가 없다.** '
              '따라서 실제 센서와 추정력/시뮬레이션력 사이 N 단위 RMSE·정확도는 계산하지 않았다. '
              '독립 축 그래프와 정상 보행 구간 Pearson 상관계수로 파형을 비교했다. '
              'z-score는 각 신호의 정상 보행 평균·표준편차를 이용한 모양 비교이며 센서 보정이 아니다.', '',
              '| 발 | 센서–준정적 추정 r (5회 평균) | 센서–시뮬레이션 r (5회 평균) |', '|---|---:|---:|']
    for leg in LEGS:
        values = []
        for kind in ('sensor_vs_quasistatic', 'sensor_vs_sim'):
            rs = [r['pearson'] for r in forces if r['foot'] == leg and r['comparison'] == kind and r['pearson'] is not None]
            values.append(float(np.mean(rs)) if rs else float('nan'))
        lines.append(f'| {leg} | {values[0]:.4f} | {values[1]:.4f} |')
    lines += ['', '[힘 파형 상관계수 요약](figure/summary_force_correlation.png)', '',
              '상관계수 평균은 시행별 기술통계이며 통계적 유의성 검정이 아니다. '
              '시뮬레이션은 같은 명령/초기화의 반복이며 실제 시행 변동성을 재현한 표본은 아니다.', '',
              '## 힘 계산의 의미와 제한', '',
              '키네마틱스만으로 힘은 결정되지 않는다. 실제 q·quaternion으로 MJCF 발 Jacobian J를 계산하고 '
              '실제 ls_tau(추정 모터 토크)를 결합했다. 중력 보정 추정은 JᵀF = g(q)−τ_est, '
              '보정 없는 비교치는 JᵀF = −τ_est를 사용한다. F는 로봇에 작용하는 월드 힘이며 Fz는 위쪽 양수다. '
              '감쇠 최소제곱 λ=0.005 m를 적용하고 음수 값을 자르거나 센서값으로 접촉 마스킹하지 않았다.', '',
              '관성 Mq̈, 코리올리, 마찰, 실제 구동계 오차를 무시한 **준정적 추정**이다. '
              '착지 충격의 정답이 아니며 음수 Fz는 추정 가정의 불일치도 반영한다. '
              '실제 발 센서의 감지 축·지연도 확인되지 않았으므로 월드 Fz와 엄밀히 같은 물리량이라고 보장할 수 없다.', '',
              '## 처리 방법', '',
              f'- 실제 LowState tick(ms)을 중복 제거하고 수신 시각과 median offset으로 정렬했다. '
              'record_start를 공통 원점으로 사용한 후 위에서 명시한 보행 위상 이동을 추가했다. 실제 센서 전송 지연은 측정하지 않았다.',
              f'- q/dq 및 힘 추정용 실제 토크는 실제 500 Hz, 시뮬레이션 200 Hz 주기에서 각각 {cutoff:g} Hz 4차 Butterworth zero-phase로 필터링한 뒤 '
              '공통 200 Hz에 보간했다. 오프라인 분석용이며 온라인 필터가 아니다.',
              '- 실제 foot sensor counts와 시뮬레이션 접촉력은 추가 저역통과 없이 공통 시간축에 보간했다. '
              '준정적 추정력은 필터링한 관절각·토크에서 계산하므로 힘 비교 신호들의 대역폭은 완전히 같지 않다.',
              '- ddq는 기록에 없으므로 필터링한 dq를 각 원래 주기에서 미분한 추정값이다. '
              '15 Hz를 넘는 충격 성분과 실제 센서 acceleration이라고 해석하면 안 된다.',
              '- 10 ms 초과 실제 tick 공백과 전후 0.2초, 전체 필터 경계 0.2초를 수치 평가에서 제외했다.',
              '- 정상 보행은 move_start+2초부터 move_stop−1초까지다. moving 전체 구간 수치도 CSV에 저장했다.',
              '- 그림은 전체 기록을 표시하고 요약 수치는 유효 구간만 사용한다. 첫 시행은 3초 확대 그림도 제공한다.', '',
              '## 산출물', '',
              '- `figure/`: 전체 25개, 첫 시행 확대 5개, 요약 2개, 정렬 진단 5개 PNG.',
              '- `synchronization/`: 시행별 offset·합의·끝 검증 JSON과 이벤트 대응 CSV.',
              '- `unaligned/`: 변경 전 원본 비교 결과.',
              '- `joint_metrics.csv`: 시행·구간·관절별 RMSE/MAE/bias/상관계수/평균/표준편차/진폭.',
              '- `force_metrics.csv`: 발별 파형 상관계수와 센서·추정력 기술통계. 절대 힘 RMSE는 공란.',
              '- `processed/`: 공통 시간축의 q/dq/ddq, 센서값, 추정력, 시뮬레이션력, Jacobian 조건수와 평가 마스크.',
              '- `summary.json`: 입력 SHA256, 모델 해시, 처리 설정과 요약 수치.', '',
              '## 시행별 그림', '']
    for entry in inputs:
        trial = entry['trial']
        lines += [f'### {trial}', '',
                  ' · '.join(f'[{q}](figure/{trial}_{q}.png)' for q in ('q', 'dq', 'ddq')), '',
                  f'[센서–준정적 추정](figure/{trial}_sensor_vs_estimated.png) · '
                  f'[센서–시뮬레이션](figure/{trial}_sensor_vs_sim.png) · '
                  f'[정렬 전후·끝 검증](figure/{trial}_synchronization.png)', '']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
