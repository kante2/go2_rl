"""Graphs, nominal LowCmd replay, and causal GRF input contracts.

No policy training, force calibration or simulator parameter optimization.
"""
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

from real2sim_converter import ROOT, JOINTS, digest, write_json, load, hold_indices, PolicyMapping, action_to_targets, restore_output_owner

sys.path.insert(0, str(ROOT / 'src/Real2Sim/scripts/offline_identification'))
from prepare_dataset import prepare
from mujoco_replay import Replay


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files if k not in ('metadata', 'meta')}


def csv_write(path, rows):
    if not rows: return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def save(fig, path):
    import matplotlib.pyplot as plt
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)


def norm_parameters(policy_dir):
    import onnx
    from onnx.numpy_helper import to_array
    model = onnx.load(policy_dir / 'policy.onnx')
    weights = {v.name: to_array(v) for v in model.graph.initializer}
    first, second = model.graph.node[:2]
    if first.op_type != 'Sub' or second.op_type != 'Div' or second.input[0] != first.output[0]:
        raise ValueError('Unsupported ONNX normalization graph; inspect before proceeding')
    mean, denominator = weights[first.input[1]].reshape(-1), weights[second.input[1]].reshape(-1)
    if mean.shape != (47,) or denominator.shape != (47,) or np.any(denominator <= 0):
        raise ValueError('Invalid ONNX normalization statistics')
    return mean, denominator


GROUPS = [('gyro', 0, 3), ('gravity', 3, 6), ('command', 6, 9), ('phase', 9, 11),
          ('q_relative', 11, 23), ('dq', 23, 35), ('previous_action', 35, 47)]


def plot_observations(data, mean, denominator, trial, output):
    import matplotlib.pyplot as plt
    real = (data['observation'][data['steady']] - mean) / denominator
    t = data['sim_t']; lo, hi = data['t'][data['steady']][[0, -1]]
    sim = (data['sim_observation'][(t >= lo) & (t <= hi)] - mean) / denominator
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    x = np.arange(47)
    for values, label, color in ((real, 'Real reconstructed', 'C0'), (sim, 'Original simulation', 'C1')):
        m, s = values.mean(axis=0), values.std(axis=0)
        axes[0].plot(x, m, color=color, label=label)
        axes[0].fill_between(x, m - s, m + s, color=color, alpha=.12)
    axes[0].set_ylabel('Normalized mean ± SD'); axes[0].legend()
    axes[1].plot(x, real.std(axis=0) / np.maximum(sim.std(axis=0), .05), color='C4')
    axes[1].set_ylabel('SD real / max(SD sim, 0.05)'); axes[1].axhline(1, color='black', lw=.6)
    rows = []
    for name, left, right in GROUPS:
        rows.append(dict(trial=trial, group=name,
            normalized_mean_discrepancy_rms=float(np.sqrt(np.mean((real.mean(axis=0)[left:right] - sim.mean(axis=0)[left:right]) ** 2))),
            real_normalized_std_mean=float(real.std(axis=0)[left:right].mean()),
            sim_normalized_std_mean=float(sim.std(axis=0)[left:right].mean())))
        for ax in axes:
            ax.axvline(left - .5, color='gray', lw=.5); ax.grid(alpha=.15)
    axes[1].set_xticks([(a + b - 1) / 2 for _, a, b in GROUPS], [n for n, _, _ in GROUPS])
    fig.suptitle(f'{trial}: deployment ONNX normalization applied exactly once\nMean/SD distribution comparison; no physical state correction')
    fig.tight_layout(); save(fig, output)
    return rows


def plot_targets(data, info, output):
    import matplotlib.pyplot as plt
    t = data['t']; mask = (t >= info['move_start_s'] + 3) & (t <= info['move_start_s'] + 6) & data['valid']
    fig, axes = plt.subplots(4, 3, figsize=(16, 11), sharex=True)
    for j, ax in enumerate(axes.flat):
        ax.plot(t[mask], data['target_logged_sdk'][mask, j], label='Recorded LowCmd', lw=1.)
        ax.plot(t[mask], data['target_onnx_rate_sdk'][mask, j], label='ONNX + reconstructed slew limit', lw=.9)
        ax.set_title(JOINTS[j].replace('_joint', '')); ax.set_ylabel('Target [rad]'); ax.grid(alpha=.2)
    for ax in axes[-1]: ax.set_xlabel('Real record time [s]')
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f'{info["trial"]}: deployment policy replay on recorded real states\nPhase fitted on start only; teacher-forced previous targets; not exact inference timestamps')
    fig.tight_layout(); save(fig, output)


def causal_grf_inputs(data, origin, query, simulated=False):
    """Causal native anti-alias -> common 50 Hz filter -> backward difference."""
    dt = .005 if simulated else .002
    if simulated:
        source = data['ls_t']; keep = np.ones(len(source), bool)
    else:
        tick = data['ls_tick'].astype(float)
        keep = np.r_[np.diff(tick) != 0, True]
        base = (tick[keep] - tick[0]) * .001
        source = base + np.median(data['ls_t'][keep] - base) - origin
    native = np.arange(source[0], source[-1], dt)
    idx, _ = hold_indices(source, native, .020)
    anti_alias = butter(4, 15., fs=1 / dt, output='sos')
    q = sosfilt(anti_alias, data['ls_q'][keep][idx], axis=0)
    dq = sosfilt(anti_alias, data['ls_dq'][keep][idx], axis=0)
    sampled, valid = hold_indices(native, query, .020)
    common = butter(2, 10., fs=50., output='sos')
    q = sosfilt(common, q[sampled], axis=0)
    dq = sosfilt(common, dq[sampled], axis=0)
    ddq = np.vstack((np.zeros((1, 12)), np.diff(dq, axis=0) / .020))
    # Exclude transients and neighborhoods downstream of missing samples.
    valid &= query >= source[0] + .6
    for index in np.flatnonzero(np.diff(source) > .010 + 1e-9):
        valid &= ~((query >= source[index]) & (query <= source[index + 1] + .6))
    label_index, label_valid = hold_indices(source, query, .020)
    labels = data['sim_foot_fz_N'][keep][label_index] if simulated else data['ls_ff'][keep][label_index]
    return np.concatenate((q, dq, ddq), axis=1), labels, valid & label_valid


def histories(values, valid, length=6):
    result = np.stack([values[i - length + 1:i + 1] for i in range(length - 1, len(values))])
    good = np.array([valid[i - length + 1:i + 1].all() for i in range(length - 1, len(values))])
    return result, good


def physical_replay(real_path, output, engine, info):
    summaries, examples = [], {}
    for horizon in (.2, 1.):
        directory = output / f'replay_{horizon:g}s'; directory.mkdir(exist_ok=True)
        dataset_path = directory / real_path.name
        prepared = prepare(real_path, dataset_path, dt=.005, horizon=horizon, stride=horizon)
        data = read_npz(dataset_path)
        raw, _ = load(real_path)
        origin = float(raw['ev_t'][raw['ev_label'] == 'record_start'][0])
        selected = []
        for i, (start, end) in enumerate(data['clip_bounds']):
            if (data['t'][start] - origin >= info['move_start_s'] + 2 and
                    data['t'][end - 1] - origin <= info['move_stop_s'] - 1):
                selected.append(i)
        errors = {key: [] for key in ('q', 'dq', 'ddq')}
        saved = {key: [] for key in ('t', 'real_q', 'real_dq', 'real_ddq', 'replay_q', 'replay_dq', 'replay_ddq', 'force_world_N', 'sensor_counts')}
        initializations = []
        filt = butter(4, 15., fs=200., output='sos')
        for i in selected:
            prediction, initialization = engine.rollout(data, i)
            start, end = data['clip_bounds'][i]
            t = data['t'][start:end] - origin
            rdq = sosfiltfilt(filt, data['dq'][start:end], axis=0)
            sdq = sosfiltfilt(filt, prediction['dq'], axis=0)
            rddq, sddq = np.gradient(rdq, .005, axis=0), np.gradient(sdq, .005, axis=0)
            errors['q'].append(prediction['q'][1:] - data['q'][start + 1:end])
            errors['dq'].append(prediction['dq'][1:] - data['dq'][start + 1:end])
            errors['ddq'].append(sddq[5:-5] - rddq[5:-5])
            values = dict(t=t, real_q=data['q'][start:end], real_dq=data['dq'][start:end], real_ddq=rddq,
                replay_q=prediction['q'], replay_dq=prediction['dq'], replay_ddq=sddq,
                force_world_N=prediction['force_world'], sensor_counts=data['ff'][start:end])
            for key in saved: saved[key].append(values[key])
            initializations.append(initialization)
            if horizon == 1. and not examples:
                examples = values
        if not selected: raise ValueError('No steady replay clips')
        np.savez_compressed(directory / (real_path.stem + '_predictions.npz'),
                            **{k: np.stack(v) for k, v in saved.items()})
        row = dict(trial=real_path.stem, horizon_s=horizon, clips=len(selected),
                   q_rmse_rad=float(np.sqrt(np.mean(np.concatenate(errors['q']) ** 2))),
                   dq_rmse_rad_s=float(np.sqrt(np.mean(np.concatenate(errors['dq']) ** 2))),
                   ddq_rmse_rad_s2=float(np.sqrt(np.mean(np.concatenate(errors['ddq']) ** 2))))
        summaries.append(row)
        write_json(directory / (real_path.stem + '_metadata.json'), dict(metrics=row,
            model=engine.metadata(), source_sha256=digest(real_path),
            initializations=initializations, prepared=prepared,
            parameters_fitted=False, initialization='real q/dq/quaternion/gyro; base height and velocity inferred',
            excluded='initial state excluded from q/dq; ddq excludes 25ms at each clip boundary',
            limitations='Every clip is reset; cannot compare these errors directly with long closed-loop rollout errors.'))
    return summaries, examples


def plot_replay(example, trial, output):
    import matplotlib.pyplot as plt
    for quantity, unit in (('q', 'rad'), ('dq', 'rad/s'), ('ddq', 'rad/s²')):
        fig, axes = plt.subplots(4, 3, figsize=(16, 11), sharex=True)
        for j, ax in enumerate(axes.flat):
            ax.plot(example['t'], example['real_' + quantity][:, j], label='Real measured' if quantity != 'ddq' else 'Real derived dq/dt')
            ax.plot(example['t'], example['replay_' + quantity][:, j], label='Nominal MuJoCo LowCmd replay', lw=.9)
            ax.set_title(JOINTS[j].replace('_joint', '')); ax.set_ylabel(unit); ax.grid(alpha=.2)
            ax.axvline(example['t'][0], color='gray', ls=':', lw=.7)
        for ax in axes[-1]: ax.set_xlabel('Real record time [s]')
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f'{trial}: one-second free physical replay, {quantity}\nState copied ONLY at clip start; nominal parameters, no fit; initial sample excluded from q/dq metrics')
        fig.tight_layout(); save(fig, output / f'{trial}_replay_{quantity}.png')
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    for foot, ax in enumerate(axes):
        one = ax.plot(example['t'][1:], example['sensor_counts'][1:, foot], label='Real sensor [counts]')
        right = ax.twinx()
        two = right.plot(example['t'][1:], example['force_world_N'][1:, foot, 2], color='C1', label='Replay GRF [N]')
        ax.set_ylabel(f'{("FR", "FL", "RR", "RL")[foot]} [counts]'); right.set_ylabel('World Fz [N]')
        ax.grid(alpha=.2)
        if foot == 0: ax.legend(one + two, [l.get_label() for l in one + two])
    axes[-1].set_xlabel('Real record time [s]')
    fig.suptitle(f'{trial}: sensor versus nominal LowCmd replay contact force\nIndependent units/axes; reset-induced contact transients possible; NOT a force accuracy validation')
    fig.tight_layout(); save(fig, output / f'{trial}_replay_force.png')


def analyze(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((out / 'conversion_summary.json').read_text())
    mapping = PolicyMapping(json.loads((args.policy_dir / 'policy.json').read_text()))
    mean, denominator = norm_parameters(args.policy_dir)
    write_json(out / 'onnx_normalization.json', dict(mean=mean.tolist(), denominator=denominator.tolist(),
        source='Sub/Div initializers from deployed ONNX; denominator already includes epsilon'))
    engine = Replay()
    obs_rows, replay_rows, target_rows, feature_records = [], [], [], []
    for entry in summary['trials']:
        trial = entry['trial']; converted = read_npz(out / 'converted' / f'{trial}.npz')
        validation = converted['validation']
        previous_target = action_to_targets(converted['previous_action'], mapping)
        entry['previous_target_rmse_validation_rad'] = float(np.sqrt(np.mean(
            (previous_target[validation] - converted['target_logged_sdk'][validation]) ** 2)))
        plot_targets(converted, entry, out / f'{trial}_policy_targets.png')
        obs_rows.extend(plot_observations(converted, mean, denominator, trial,
                                          out / f'{trial}_normalized_observations.png'))
        mask = converted['validation']
        for j, name in enumerate(JOINTS):
            target_rows.append(dict(trial=trial, joint=name,
                held_out_target_rmse_rad=float(np.sqrt(np.mean((converted['target_onnx_rate_sdk'][mask, j] - converted['target_logged_sdk'][mask, j]) ** 2)))))
        real_path = args.data_dir / 'real' / f'{trial}.npz'
        rows, example = physical_replay(real_path, out, engine, entry)
        from full_record_replay import full_replay
        full_replay(real_path, out / 'replay_full', engine)
        replay_rows.extend(rows); plot_replay(example, trial, out)
        real, _ = load(real_path); sim, _ = load(args.data_dir / 'sim' / f'{trial}.npz')
        origin = float(real['ev_t'][real['ev_label'] == 'record_start'][0])
        for domain, raw in (('real', real), ('sim', sim)):
            values, labels, valid = causal_grf_inputs(raw, origin, converted['t'], domain == 'sim')
            history, good = histories(values, valid)
            feature_records.append(dict(trial=trial, domain=domain, history=history[good],
                t=converted['t'][5:][good], labels=labels[5:][good], current=values[5:][good]))
        print(f'{trial}: policy graphs, nominal dynamics replay, causal GRF histories done', flush=True)
    training = [r['current'] for r in feature_records if r['domain'] == 'real' and r['trial'].endswith(('_1', '_2', '_3'))]
    x = np.concatenate(training); feature_mean = x.mean(axis=0); feature_std = np.maximum(x.std(axis=0), 1e-6)
    feature_dir = out / 'grf_inputs'; feature_dir.mkdir(exist_ok=True)
    for record in feature_records:
        trial, domain = record['trial'], record['domain']
        np.savez_compressed(feature_dir / f'{trial}_{domain}.npz', t=record['t'],
            history_physical=record['history'], history_normalized=(record['history'] - feature_mean) / feature_std,
            sensor_counts=record['labels'] if domain == 'real' else np.empty((0, 4)),
            sim_force_N=record['labels'] if domain == 'sim' else np.empty((0, 4)))
    contract = dict(schema='go2-grf-input-contract-v1', trained_predictor=False, reward_modified=False,
        feature_order='q_sdk[12], dq_sdk[12], ddq_sdk[12]; absolute q, SI units',
        joint_order=JOINTS, frame_rate_hz=50, history_frames=6, history_shape=[6, 36],
        preprocessing='native causal sample-hold -> 15Hz Butterworth4 -> causal 50Hz sample-hold -> common 10Hz Butterworth2; backward dq difference /0.02',
        common_filter_sos=butter(2, 10, fs=50, output='sos').tolist(),
        normalization_mean=feature_mean.tolist(), normalization_std=feature_std.tolist(),
        normalization_fit='real trials 0.2_1/2/3 current frames only; same frozen statistics for real/sim',
        split=dict(train=['0.2_1', '0.2_2', '0.2_3'], validation=['0.2_4'], test=['0.2_5']),
        real_label='UNCALIBRATED SENSOR COUNTS; not GRF in newtons', sim_label='world vertical contact force N',
        gap_policy='exclude 0.6s after source gaps; all six history frames must be valid; no cross-trial history',
        limitations=['Missing actual sensor delays and base velocity ground truth.',
                     'Filtering reduces high-frequency impact content; 50Hz cannot certify sub-step impact peaks.',
                     'Native anti-alias discretizations differ at 500/200Hz; common downstream operator is identical.',
                     'No force calibration, predictor training, dynamics identification or policy reward change performed.'])
    write_json(feature_dir / 'contract.json', contract)
    csv_write(out / 'observation_metrics.csv', obs_rows)
    csv_write(out / 'target_metrics.csv', target_rows)
    csv_write(out / 'replay_metrics.csv', replay_rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in zip(axes, ('q_rmse_rad', 'dq_rmse_rad_s', 'ddq_rmse_rad_s2'), ('q [rad]', 'dq [rad/s]', 'ddq [rad/s²]')):
        for horizon, color in ((.2, 'C0'), (1., 'C1')):
            rows = [r for r in replay_rows if r['horizon_s'] == horizon]
            ax.plot(range(1, 6), [r[key] for r in rows], marker='o', color=color, label=f'{horizon:g}s free replay')
        ax.set_title(title); ax.set_xlabel('Trial'); ax.set_ylabel('RMSE'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Nominal dynamics under real LowCmd; reset at each clip, no physical parameter fitting')
    fig.tight_layout(); save(fig, out / 'summary_replay_errors.png')
    summary.update(replay_metrics=replay_rows, observation_metrics=obs_rows,
                   predictor_contract='grf_inputs/contract.json', analysis_scope='interface validation + nominal dynamics diagnosis; no fitted Real2Sim correction')
    write_json(out / 'analysis_summary.json', summary)
    write_report(out, summary)
    from force_diagnostics import write_diagnostics
    write_diagnostics(out)
    restore_output_owner(out)
    print(f'Analysis: {out / "report.md"}', flush=True)


def write_report(out, summary):
    verified = summary['verification']
    lines = ['# 배포 파일 기반 Real2Sim converter 분석', '',
        '## 결과의 의미', '',
        '**배포 관측·action 변환을 재사용한 실제 로그 재구성과 명목 동역학 진단이다. '
        '물리 파라미터를 보정하거나 GRF predictor/정책을 학습한 결과가 아니다.**', '',
        'policy.onnx(정규화 포함), policy.json(순서/기본 자세/scale/PD), parity.npz와 samples.json(변환 검증), '
        '실기 real_control.py의 PolicyMapping.observation() 및 JointTargetRamp를 재사용했다.', '',
        '## 배포 규약 검증', '',
        f'- 검증 프레임: {verified["frames"]}개.',
        f'- ONNX–저장 PT 출력 최대 절대 오차: {verified["onnx_torch_max_error"]:.3e}.',
        f'- 관측 매핑 재구성 최대 절대 오차: {verified["observation_mapping_max_error"]:.3e}.',
        '- ONNX 정규화는 모델 안에서 한 번만 실행한다. 그림용 통계는 같은 ONNX Sub/Div 상수에서 추출했다.', '',
        '## 실제 로그에서 정책 재생', '',
        '50Hz 기준 시각마다 과거에 수신된 최신 LowState·속도 명령을 사용했다. 이전 action은 20ms 이전 '
        'LowCmd 목표각에서 (q_target−q_default)/action_scale로 역산했다. 미래 센서는 입력에 쓰지 않았다.', '',
        '실기 위상 원점이 저장되지 않아 시작 정상 보행 5초에서 30개 위상 후보를 탐색했다. '
        '각 후보의 ONNX raw 목표각과 실제 LowCmd 차이만 사용하고 끝 5초는 검증에 남겼다. '
        '이는 실제 추론 시점/필터 내부 상태를 완전히 복원한 것이 아니다. slew limiter는 관측된 이전 목표각을 '
        '사용한 teacher-forced 한 스텝 근사다. 실제 raw action의 정답은 기록되어 있지 않다.', '',
        '| 시행 | 추정 위상 offset (s) | 미사용 ONNX 목표각 RMSE (rad) | 이전 목표각 유지 RMSE (rad) | q 역변환 최대 오차 (rad) |',
        '|---|---:|---:|---:|---:|']
    for row in summary['trials']:
        baseline = row.get('previous_target_rmse_validation_rad')
        baseline_text = f'{baseline:.6f}' if baseline is not None else 'N/A'
        lines.append(f'| {row["trial"]} | {row["phase_offset_s"]:.2f} | {row["rate_target_rmse_validation_rad"]:.6f} | {baseline_text} | {row["q_roundtrip_max_error_rad"]:.3e} |')
    lines += ['', 'q의 왕복 변환 오차가 작다는 것은 관절 순서·offset·scale이 일관된다는 뜻이다. '
              '실제 관절 궤적을 시뮬레이션 궤적으로 보정했다는 뜻은 아니다. '
              '원래 저장된 시뮬레이션 관측에 ONNX를 실행해 기존 PT action과 일치하는지도 모든 프레임에서 확인했다.', '',
              '## 실제 LowCmd의 물리적 재생', '',
              '원래 LowCmd의 q/dq/Kp/Kd/feedforward를 명목 MuJoCo에 입력했다. 시작 상태만 실제 q/dq/자세/각속도로 '
              '설정하고 이후 자유 적분했다. 몸체 높이·선속도는 발 기하와 지지발 no-slip 조건에서 추정했다. '
              '관절 gain·질량·마찰 등 파라미터는 학습/최적화하지 않았다.', '',
              '| 시행 | horizon (s) | clips | q RMSE (rad) | dq RMSE (rad/s) | ddq RMSE (rad/s²) |',
              '|---|---:|---:|---:|---:|---:|']
    for row in summary['replay_metrics']:
        lines.append(f'| {row["trial"]} | {row["horizon_s"]:g} | {row["clips"]} | {row["q_rmse_rad"]:.5f} | {row["dq_rmse_rad_s"]:.5f} | {row["ddq_rmse_rad_s2"]:.3f} |')
    averages = {}
    for horizon in (.2, 1.):
        rows = [r for r in summary['replay_metrics'] if r['horizon_s'] == horizon]
        averages[horizon] = {key: float(np.mean([r[key] for r in rows])) for key in
                            ('q_rmse_rad', 'dq_rmse_rad_s', 'ddq_rmse_rad_s2')}
    lines += ['', '### 현재 수치의 해석', '',
              f'5회 시행별 RMSE의 단순 평균은 0.2초에서 q={averages[.2]["q_rmse_rad"]:.5f} rad, '
              f'dq={averages[.2]["dq_rmse_rad_s"]:.4f} rad/s이고, 1초에서 '
              f'q={averages[1.]["q_rmse_rad"]:.5f} rad, dq={averages[1.]["dq_rmse_rad_s"]:.4f} rad/s다. '
              '예측 구간을 늘리면 모든 시행에서 오차가 커졌다. 배포 관측/모델 변환이 일치해도 '
              '명목 구동계·접촉 동역학과 추정 초기 상태만으로 실제 운동이 정확히 재현되지는 않는다. '
              '다만 두 horizon의 clip 구성이 같지 않으므로 순수 시간 누적 효과만 분리한 실험은 아니다.', '',
              '정규화 관측 평균 차이는 다음과 같다. 각 그룹 값은 ONNX 정규화 좌표에서 '
              '실제−시뮬레이션 평균 차이의 RMS를 계산하고 5회 평균한 것이다. 물리 단위 오차나 백분율이 아니다.', '',
              '| 관측 그룹 | 정규화된 평균 차이 RMS |', '|---|---:|']
    for group, _, _ in GROUPS:
        rows = [r for r in summary['observation_metrics'] if r['group'] == group]
        value = float(np.mean([r['normalized_mean_discrepancy_rms'] for r in rows]))
        lines.append(f'| {group} | {value:.4f} |')
    lines += ['', '속도 명령 평균은 일치하지만 자세에 따른 중력 방향, 상대 관절각, 이전 action 등의 '
              '관측 분포는 다르다. 즉 단순 scale 수정만으로 해결할 문제라고 보기 어렵다. '
              '표의 평균 차이가 작아도 분산·접촉 순간 오차가 작다는 의미는 아니므로 관측 분포 그림을 함께 확인해야 한다.', '',
              '초기 복사 상태는 q/dq 오차에서 제외했고 ddq는 양쪽 dq를 15Hz zero-phase 필터링해 미분한 뒤 '
              '구간 양끝 25ms를 제외했다. 매 clip 초기화의 도움을 받으므로 기존 장시간 폐루프 그래프와 '
              '직접 비교해 물리 모델이 개선됐다고 주장할 수 없다. 1초에서 오차가 커지면 누적 동역학/초기화 오차의 증거다.', '',
              '[재생 오차 요약](summary_replay_errors.png)', '',
              '발 힘 그래프의 실제 센서는 counts, MuJoCo 힘은 N이다. 단위 보정이 없어 힘 RMSE는 계산하지 않았다. '
              '초기 발 접촉 복원 오차가 인공적인 force transient를 만들 수 있으므로 충격 피크 정답으로 쓰면 안 된다.', '',
              '## GRF predictor에 반영할 방법', '',
              '1. **정책 변환과 predictor 변환을 구분한다.** 정책 47차원 관측/ONNX 정규화와 '
              'GRF 입력 [q,dq,ddq] 이력은 별도 계약이다. policy.json의 관절 순서를 재사용하고 predictor에는 '
              '실제 학습 구간에서 구한 독립적인 고정 통계를 사용한다.',
              '2. **동일한 causal 입력 처리로 학습/보상을 연결한다.** `grf_inputs/`에 Go2 12관절 '
              '6프레임 × 36차원 이력(216개 값)을 내보냈다. native anti-alias 후 공통 50Hz causal 필터와 '
              'backward difference로 ddq를 만들었다. 시각화의 zero-phase 미분과 다르다.',
              '3. **시험 데이터 누수를 막는다.** 실제 1~3회차로 입력 통계만 계산했고, 4회차는 validation, '
              '5회차는 test로 구분했다. 아직 predictor 성능을 측정한 것은 아니다. 모든 파일은 0.2m/s baseline이므로 '
              '새 속도/새로운 저충격 보행으로의 일반화를 보장하지 않는다.',
              '4. **힘 라벨을 확보한다.** 실제 라벨은 보정되지 않은 센서 counts다. N 단위 predictor라고 '
              '부르기 전에 발별 영점·하중 보정이 필요하다. 관절 토크 기반 준정적 추정치는 보조 기준이며 '
              '독립적인 힘 센서 정답을 대체하지 않는다.',
              '5. **보상 없이 먼저 검증한다.** 실제 미사용 시행에서 접촉/피크/충격량을 확인하고, '
              '시뮬레이션에서도 같은 predictor의 추정력과 물리엔진 접촉력을 별도 비교한다. '
              '명목 LowCmd replay 오차가 큰 구간은 구동계·초기화·물리 식별을 먼저 개선한다.',
              '6. **물리 제약을 추가할 때 모델 오차를 접촉력으로 몰아넣지 않는다.** '
              'QuietWalk처럼 Mq̈+Cq̇+G−τ_eff−JᵀF 잔차를 사용할 수 있지만, floating-base 상태의 부족과 '
              '미식별 구동계 오차가 남아 있다. 보정되지 않은 모델을 강한 정답으로 고정하면 '
              '그 오차를 predictor가 잘못된 힘으로 설명할 수 있다. 토크/목표각 같은 사용 가능한 입력의 추가도 별도 검증할 가치가 있다.', '',
              '## policy 보상에 연결하는 제안', '',
              '보정된 N 단위 라벨로 predictor를 학습·검증한 뒤 predictor와 전처리 통계를 고정한다. '
              'Go2 네 발의 기본 보상 형태는 다음과 같다.', '',
              r'$$r_{impact}=-\alpha\sum_{i=1}^{4}(\hat F_{z,i}/F_{ref})^2$$', '',
              '`F_ref`는 고정된 물리적 힘 기준(예: 명목 mg/4)이며 모델마다 다시 적합하지 않는다. '
              '배포용 predictor를 reward 계산에만 사용하고 PPO 가치 critic과는 구분한다. '
              '처음에는 작은 α를 사용하며 속도 추종·안정성 보상과 함께 학습한다.', '',
              '예측 힘은 줄고 실제 시뮬레이션 피크가 증가하는 reward exploitation을 반드시 감시한다. '
              '물리엔진 피크/충격량, 스윙 가짜 힘, predictor 불확실성/OOD를 독립적으로 기록한다. '
              '최종 검증은 실제 센서와 소음으로 수행한다. 50Hz/저역통과 이력은 고주파 착지 피크를 완전히 '
              '포착하지 못하므로 고대역 힘 측정을 추가해야 한다.', '',
              '**현재 배포 파일은 표현 불일치를 확인하는 데 충분하지만 현실의 동역학을 자동 역변환하지 않는다. '
              '이 결과만으로 GRF 보상을 실기에 신뢰할 수 있다고 결론 내릴 수 없다.**', '',
              '## 파일', '',
              '- `converted/`: 관측, ONNX action, 기록/재구성 목표각, 위상 적합·미사용 평가 수치.',
              '- `replay_0.2s/`, `replay_1s/`: 원본 LowCmd와 명목 모델 기반 자유 예측.',
              '- `grf_inputs/contract.json`: predictor 입력 처리·정규화·단위·분할 계약.',
              '- `asset_verification.json`, `analysis_summary.json`, 각 CSV: 해시와 검증/분석 결과.', '']
    for entry in summary['trials']:
        trial = entry['trial']
        lines += [f'### {trial}', '', f'[정책 목표각 재생]({trial}_policy_targets.png) · '
            f'[정규화 관측 분포]({trial}_normalized_observations.png)', '',
            ' · '.join(f'[LowCmd replay {q}]({trial}_replay_{q}.png)' for q in ('q', 'dq', 'ddq', 'force')), '']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
