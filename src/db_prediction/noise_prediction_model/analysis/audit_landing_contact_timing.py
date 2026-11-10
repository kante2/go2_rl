#!/usr/bin/env python3
"""Audit force-threshold onset vs estimated foot stopping, without retiming."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/low_noise_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'preprocessing'))
from go2_kinematics import Go2Kinematics

BASE = ROOT / 'noise_prediction_model/results/analysis/landing_dynamics'
REL = np.arange(-50, 102, 2)


def first_time(mask, times):
    return np.where(mask.any(axis=1), times[np.argmax(mask, axis=1)], np.nan)


def timing(events, samples):
    identifiers = samples.drop_duplicates('event_id').event_id
    events = events.set_index('event_id').loc[identifiers].reset_index()
    if len(samples) != len(events) * len(REL):
        raise ValueError('Unexpected sample count')
    times = samples.relative_time_ms.to_numpy().reshape(-1, len(REL))
    if not np.array_equal(times, np.broadcast_to(REL, times.shape)):
        raise ValueError('Unexpected ordering of event-relative times')
    velocity = samples.vz.to_numpy().reshape(-1, len(REL))
    force = samples.force.to_numpy().reshape(-1, len(REL))
    near = abs(velocity) <= .1
    post = (REL >= 0) & (REL <= 40)
    events['vz_at_force_on_mps'] = velocity[:, REL == 0].ravel()
    events['force_at_force_on'] = force[:, REL == 0].ravel()
    events['first_abs_vz_le_0p1_after_on_ms'] = first_time(near[:, post], REL[post])
    events['first_vz_nonnegative_after_on_ms'] = first_time(velocity[:, post] >= 0, REL[post])
    stable = np.full(len(events), np.nan)
    # Six grid points span 10ms; grid evidence, not continuous zero velocity.
    for i in np.flatnonzero((REL >= 0) & (REL <= 90)):
        accepted = near[:, i:i+6].all(axis=1) & np.isnan(stable)
        stable[accepted] = REL[i]
    events['first_abs_vz_le_0p1_for_10ms_ms'] = stable
    confirmation = (events.confirmed_time_abs - events.contact_time_abs).to_numpy() * 1000
    grid = np.rint(confirmation / 2).astype(int) + 25
    if np.any((grid < 0) | (grid >= len(REL))) or not np.allclose(REL[grid], confirmation, atol=1e-5):
        raise ValueError('Confirmation outside event grid')
    events['confirmation_delay_ms_audit'] = confirmation
    events['vz_at_confirmation_mps'] = velocity[np.arange(len(events)), grid]
    return events, velocity, force


def summarize(events, keys):
    return events.groupby(keys, as_index=False).agg(
        n_events=('event_id', 'size'), command_vx=('command_vx', 'first'),
        synchronization_status=('synchronization_status', 'first'),
        median_vz_at_on_mps=('vz_at_force_on_mps', 'median'),
        fraction_abs_vz_at_on_gt_0p1=('vz_at_force_on_mps', lambda x: x.abs().gt(.1).mean()),
        median_force_at_on=('force_at_force_on', 'median'),
        median_first_near_zero_ms=('first_abs_vz_le_0p1_after_on_ms', 'median'),
        fraction_no_near_zero_in_40ms=('first_abs_vz_le_0p1_after_on_ms', lambda x: x.isna().mean()),
        median_first_nonnegative_ms=('first_vz_nonnegative_after_on_ms', 'median'),
        median_first_stable_10ms_ms=('first_abs_vz_le_0p1_for_10ms_ms', 'median'),
        fraction_no_stable_10ms=('first_abs_vz_le_0p1_for_10ms_ms', lambda x: x.isna().mean()),
        median_confirmation_delay_ms=('confirmation_delay_ms_audit', 'median'),
        median_vz_at_confirmation_mps=('vz_at_confirmation_mps', 'median'))


def component_audit(events, output):
    model = Go2Kinematics()
    components, checks = [], []
    selected_times = (-2, 0, 6, 10, 40, 80)
    for trial, local in events.groupby('trial'):
        frame = pd.read_csv(ROOT / f'preprocessed/1007/train/{trial}.csv')
        times = frame.time_abs.to_numpy()
        rotation = Rotation.from_quat(frame[[f'ls_quaternion_{a}' for a in 'wxyz']].to_numpy()[:, [1, 2, 3, 0]])
        gyro = frame[[f'ls_gyro_{a}' for a in 'xyz']].to_numpy()
        vbody = rotation.apply(frame[[f'sp_vel_{a}' for a in 'xyz']].to_numpy())[:, 2]
        for leg in ('FR', 'FL', 'RR', 'RL'):
            k = leg.lower()
            q = frame[[f'ls_q_{k}_{j}' for j in ('hip', 'thigh', 'calf')]].to_numpy()
            dq = frame[[f'ls_dq_{k}_{j}' for j in ('hip', 'thigh', 'calf')]].to_numpy()
            p, jac = model.position_jacobian(k, q)
            joint = rotation.apply(np.einsum('nij,nj->ni', jac, dq))[:, 2]
            angular = rotation.apply(np.cross(gyro, p))[:, 2]
            for event in local[local.leg == leg].itertuples():
                center = np.searchsorted(times, event.contact_time_abs - 1e-8)
                if abs(times[center] - event.contact_time_abs) > 1e-7:
                    raise AssertionError('Onset timestamp mismatch')
                for ms in selected_times:
                    i = center + ms // 2
                    components.append(dict(event_id=event.event_id, trial=trial, leg=leg, relative_time_ms=ms,
                                           body_vz_mps=vbody[i], gyro_vz_mps=angular[i], joint_vz_mps=joint[i],
                                           total_vz_mps=vbody[i] + angular[i] + joint[i]))
        # Native LowState clock consistency check. Position differentiation is
        # another estimator from the same joints, not independent ground truth.
        with np.load(ROOT / f'data/1007/train/{trial}.npz', allow_pickle=False) as raw:
            t = raw['ls_t']; q = raw['ls_q']; dq = raw['ls_dq']; ff = raw['ls_ff']
            move = dict(zip(raw['ev_label'].astype(str), raw['ev_t']))
        for index, leg in enumerate(('FR', 'FL', 'RR', 'RL')):
            k = index * 3
            p, jac = model.position_jacobian(leg.lower(), q[:, k:k+3])
            reported = np.einsum('nij,nj->ni', jac, dq[:, k:k+3])[:, 2]
            derivative = np.full(len(t), np.nan)
            span = t[6:] - t[:-6]
            derivative[3:-3] = (p[6:, 2] - p[:-6, 2]) / span
            good = (t >= move['move_start']) & (t <= move['move_stop']) & np.isfinite(derivative)
            good[3:-3] &= span <= .020
            contacts = local[local.leg == leg].contact_time_abs.to_numpy()
            source = np.searchsorted(t, contacts, side='right') - 1
            valid_contact = good[source]
            changes = np.flatnonzero(np.r_[False, np.diff(ff[:, index]) != 0])
            changes = changes[(t[changes] >= move['move_start']) & (t[changes] <= move['move_stop'])]
            checks.append(dict(trial=trial, leg=leg, n_valid_native_samples=int(good.sum()),
                               joint_vz_vs_position_derivative_mae_mps=float(np.mean(abs(reported[good] - derivative[good]))),
                               median_joint_vz_at_on_mps=float(np.median(reported[source[valid_contact]])),
                               median_position_derivative_at_on_mps=float(np.median(derivative[source[valid_contact]])),
                               force_value_change_interval_median_ms=float(np.median(np.diff(t[changes])) * 1000)))
        print(f'Audited {trial}', flush=True)
    components = pd.DataFrame(components)
    components.to_csv(output / 'velocity_components.csv', index=False)
    components.groupby(['trial', 'relative_time_ms'])[['body_vz_mps', 'gyro_vz_mps', 'joint_vz_mps', 'total_vz_mps']].median().to_csv(
        output / 'velocity_components_by_trial.csv')
    checks = pd.DataFrame(checks)
    checks.to_csv(output / 'native_joint_consistency.csv', index=False)
    return components, checks


def plot(events, velocity, force, output):
    groups = [np.flatnonzero(events.trial.eq(t)) for t in events.trial.unique()]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, layout='constrained')
    for ax, values, label, color in zip(axes[:2], (force, velocity),
                                      ('Foot force (sensor units)', 'Estimated foot v_z (m/s)'), ('tab:blue', 'crimson')):
        profiles = np.array([np.median(values[g], axis=0) for g in groups])
        q = np.percentile(profiles, [25, 50, 75], axis=0)
        ax.fill_between(REL, q[0], q[2], color=color, alpha=.2)
        ax.plot(REL, q[1], color=color, lw=2, label='Median across trial medians')
        ax.set_ylabel(label); ax.legend(loc='lower right', fontsize=8)
    axes[1].axhspan(-.1, .1, color='green', alpha=.1, label='Near-zero criterion: |v_z| <= 0.1m/s')
    axes[1].axhline(0, color='.5', lw=.7)
    near_fraction = np.array([np.mean(abs(velocity[g]) <= .1, axis=0) for g in groups])
    q = np.percentile(near_fraction, [25, 50, 75], axis=0)
    axes[2].fill_between(REL, q[0], q[2], color='green', alpha=.18)
    axes[2].plot(REL, q[1], color='green', lw=2)
    axes[2].set(ylabel='Fraction of landings\nwith |v_z| <= 0.1m/s', ylim=(0, 1), xlabel='Time from first sustained force-threshold crossing (ms)')
    for ax in axes:
        ax.axvline(0, color='black', ls='--', lw=1.3, label='First force threshold crossing')
        ax.axvline(6, color='tab:orange', ls=':', lw=1.2)
        ax.grid(alpha=.2); ax.set_xlim(-50, 100)
    axes[0].set_title('Black dashed = original force-threshold onset (0ms); orange dotted = +6ms reference\n'
                      'Force-confirmation usually occurs after onset; no timestamps have been shifted')
    fig.suptitle(f'Force-detected onset is different from foot stopping | all {events.trial.nunique()} train trials\n'
                 'Bands = IQR across trials; near-zero threshold is a diagnostic choice, not contact ground truth', fontsize=12)
    fig.savefig(output / 'contact_onset_vs_velocity.png', dpi=165); plt.close(fig)


def main():
    output = BASE / 'contact_timing_audit'
    output.mkdir(exist_ok=True)
    source_events = pd.read_csv(BASE / 'landing_event_metrics.csv')
    samples = pd.read_csv(BASE / 'landing_aligned_samples.csv', usecols=['event_id', 'relative_time_ms', 'force', 'vz'])
    events, velocity, force = timing(source_events, samples)
    columns = ['event_id', 'trial', 'leg', 'command_vx', 'synchronization_status',
               'vz_at_force_on_mps', 'force_at_force_on', 'first_abs_vz_le_0p1_after_on_ms',
               'first_vz_nonnegative_after_on_ms', 'first_abs_vz_le_0p1_for_10ms_ms',
               'confirmation_delay_ms_audit', 'vz_at_confirmation_mps']
    events[columns].to_csv(output / 'contact_timing_events.csv', index=False)
    by_trial = summarize(events, ['trial'])
    by_trial.to_csv(output / 'contact_timing_by_trial.csv', index=False)
    summarize(events, ['trial', 'leg']).to_csv(output / 'contact_timing_by_trial_leg.csv', index=False)
    components, checks = component_audit(events, output)
    plot(events, velocity, force, output)
    result = dict(n_trials=int(events.trial.nunique()), n_events=len(events),
                  median_vz_at_force_on_mps=float(events.vz_at_force_on_mps.median()),
                  fraction_abs_vz_at_force_on_gt_0p1=float(events.vz_at_force_on_mps.abs().gt(.1).mean()),
                  median_first_near_zero_ms=float(events.first_abs_vz_le_0p1_after_on_ms.median()),
                  median_first_nonnegative_ms=float(events.first_vz_nonnegative_after_on_ms.median()),
                  median_first_stable_10ms_ms=float(events.first_abs_vz_le_0p1_for_10ms_ms.median()),
                  fraction_no_near_zero_in_40ms=float(events.first_abs_vz_le_0p1_after_on_ms.isna().mean()),
                  median_vz_at_confirmation_mps=float(events.vz_at_confirmation_mps.median()),
                  fraction_near_zero_at_confirmation=float(events.vz_at_confirmation_mps.abs().le(.1).mean()),
                  median_native_joint_position_velocity_mae_mps=float(checks.joint_vz_vs_position_derivative_mae_mps.median()),
                  components_at_on=components[components.relative_time_ms == 0][['body_vz_mps', 'gyro_vz_mps', 'joint_vz_mps', 'total_vz_mps']].median().to_dict(),
                  diagnostic_threshold_mps=.1, no_timestamps_changed=True, no_velocity_clamping=True,
                  caveat='Native q position derivative and Jdq use the same joints, not independent contact or velocity truth.')
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    report = f'''# 접촉력 검출 시작과 발 정지 시점의 진단

## 확인된 수치

{events.trial.nunique()}개 train trial, {len(events):,}개 착지를 모두 확인했다. 기존 시각·속도·Spearman 결과는 변경하지 않았다.

|지표|전체 이벤트 중앙값|
|---|---:|
|힘 임계값 첫 교차 0ms의 추정 수직 속도|{result['median_vz_at_force_on_mps']:+.3f}m/s|
|처음으로 추정 속도가 ±0.1m/s 안에 들어온 시점|+{result['median_first_near_zero_ms']:.0f}ms|
|처음으로 추정 속도가 0 이상이 된 시점|+{result['median_first_nonnegative_ms']:.0f}ms|
|±0.1m/s를 10ms 동안 유지하기 시작한 시점|+{result['median_first_stable_10ms_ms']:.0f}ms|
|검출 확인 시점의 추정 수직 속도|{result['median_vz_at_confirmation_mps']:+.3f}m/s|

0ms에 추정 속도의 절댓값이 0.1m/s보다 큰 이벤트는 {100*result['fraction_abs_vz_at_force_on_gt_0p1']:.1f}%다.
처음 근접·안정 시점의 중앙값은 해당 조건을 만족한 이벤트에 대한 값이다. 처음 ±0.1m/s에 들어왔다고 계속 정지하는 것은 아니다.
확인 시점(+6ms)에 실제로 ±0.1m/s 안에 있는 비율은 {100*result['fraction_near_zero_at_confirmation']:.1f}%뿐이다. 중앙값이 0에 가깝다고 모든 발이 그때 멈춘 것은 아니다.
40ms 안에 한 번도 근접하지 않은 비율은 {100*result['fraction_no_near_zero_in_40ms']:.2f}%다.
10ms 안정 조건은 2ms 간격 여섯 표본으로 평가했으며 독립적인 실제 정지 판정이 아니다.

## 왜 검은 선에서 속도가 0이 아닌가

현재 검은 선은 force >= 10이 지속되기 시작한 첫 교차를 의미한다. 이후 6ms 이상 증거가 유지되어야 접촉을 확정하지만, 표시 시각은 첫 교차로 소급했다.
따라서 검은 선은 접촉력 검출 시작의 대리지표이며 정지 완료 시각이나 실제 최초 접촉의 정답이 아니다.
확인 시점을 0ms로 바꾸면 그림의 속도가 0에 가까워 보일 수 있지만, 그것으로 물리적인 정확도가 검증되는 것은 아니다.

실제 충돌의 유한한 감속·발바닥 변형, force 및 q/dq의 내부 필터/시각 차이, 기구학·몸체 속도 추정 오차를 현재 기록만으로 분리하지 못했다.
컴플라이언스가 있는 접촉에서 접근 속도가 0으로 바뀌는 데 유한한 시간이 걸릴 수 있다는 일반적 설명은
[Drake contact modeling](https://drake.mit.edu/doxygen_cxx/group__compliant__contact.html)을 참고한다.
이는 본 Go2 데이터에서 변형이 주원인임을 검증한 결과는 아니다.

## 코드·원본 일관성 확인

각 발의 속도를 관절운동, 몸체 회전, 몸체 선속도 항으로 분해했다. 0ms 항별 중앙값은 관절운동
{result['components_at_on']['joint_vz_mps']:+.3f}, 회전 {result['components_at_on']['gyro_vz_mps']:+.3f}, 몸체 선속도
{result['components_at_on']['body_vz_mps']:+.3f}m/s다. 항별 중앙값을 합친 값은 전체 속도 중앙값과 같을 필요가 없다.
0ms의 큰 음의 값은 몸체 선속도 항만으로 생긴 현상이 아니다.

원본 LowState 시각에서 q로 구한 발의 몸체 상대 위치를 차분하고 J(q)dq의 상대 수직 속도와 비교했다.
trial·발별 MAE의 중앙값은 {result['median_native_joint_position_velocity_mae_mps']:.3f}m/s다.
이것은 기구학 구현과 q/dq가 대체로 일관된다는 진단이며 지면 좌표계 속도나 실제 접촉 시각의 독립 검증은 아니다.
force와 q/dq는 같은 LowState 메시지에 있지만 각 필드의 실제 센서 획득·필터 시각까지 같다는 보증은 없다.
raw force의 값 변경 간격도 저장했으나, 값 변경 빈도만으로 실제 센서 표본율이나 내부 지연을 확정하지 않는다.

## 기존 해석에 미치는 영향

기존 착지 직전 속도는 정확히는 힘 임계값 교차 직전의 추정 속도다. 0ms 이전부터 감속이 시작되는 파형이 있어
이를 물리적인 최초 충돌 직전 속도라고 확정하면 안 된다. 이전 Spearman은 현 검출기준 주변의 기술 통계로 유지한다.
속도를 접촉 후 강제로 0으로 만들거나 모든 신호를 속도 영점에 맞추면 오류를 가릴 수 있어 시행하지 않았다.
먼저 접촉력 상승 시작, 검출 확인, 추정 속도 첫 영점 근접, 안정 구간을 별도 시점으로 다루는 것이 필요하다.
이 진단은 오디오 동기화와 구분되며 오디오 시각 이동만으로 force–velocity 차이가 해결되는 것은 아니다.

## 파일

- contact_onset_vs_velocity.png: 전체 trial 파형 및 ±0.1m/s 근접 비율. 검은 선 0ms, 주황 선 +6ms 참고선.
- contact_timing_by_trial.csv / contact_timing_by_trial_leg.csv: trial별 및 발별 진단.
- contact_timing_events.csv: 각 착지의 검출 시작·확인·근접·안정 시점.
- velocity_components.csv / velocity_components_by_trial.csv: 속도 구성항.
- native_joint_consistency.csv: 원본 LowState q 위치 차분과 Jdq 비교.
- summary.json: 전체 수치 및 판정 기준.

재실행: `.venv/bin/python noise_prediction_model/analysis/audit_landing_contact_timing.py`
'''
    (output / 'report.md').write_text(report, encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
