#!/usr/bin/env python3
"""Build a read-only web dataset from existing paired logs only; never execute a simulator."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent
DEFAULT_DATASET = '1116_RL_base'
TRIALS = tuple(f'0.2_{i}' for i in range(1, 6))
sys.path.insert(0, str(PROJECT / 'scripts'))
from compare_real_sim import load, prepare_pair, JOINTS, LEGS
from phase_synchronization import estimate_alignment, apply_alignment
from deployment_mapping import load_contract, apply_contract


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(real, sim, valid):
    a, b = np.asarray(real)[valid], np.asarray(sim)[valid]
    if not len(a): return dict(n=0, rmse=None, mae=None, bias=None, correlation=None)
    error = b - a
    r = float(np.corrcoef(a, b)[0, 1]) if min(a.std(), b.std()) > 1e-10 else None
    return dict(n=len(a), rmse=float(np.sqrt(np.mean(error ** 2))),
                mae=float(np.mean(abs(error))), bias=float(error.mean()), correlation=r)


def rolling_rmse(real, sim, valid, samples=100):
    """Trailing window; invalid samples do not enter either sum or count."""
    error = np.asarray(sim) - np.asarray(real)
    good = np.asarray(valid, bool) & np.isfinite(error).all(axis=1)
    sums = np.vstack((np.zeros((1, error.shape[1])), np.cumsum(np.where(good[:, None], error ** 2, 0.), axis=0)))
    counts = np.r_[0, np.cumsum(good)]
    end = np.arange(len(error)) + 1; start = np.maximum(0, end - samples)
    count = counts[end] - counts[start]
    mean_squared = np.full_like(error, np.nan, dtype=float)
    np.divide(sums[end] - sums[start], count[:, None], out=mean_squared, where=count[:, None] > 0)
    return np.sqrt(mean_squared), np.sqrt(np.mean(mean_squared, axis=1))


def nullable(values):
    values = np.asarray(values)
    result = np.round(values, 6).astype(object)
    result[~np.isfinite(values)] = None
    return result.tolist()


def series(data, kind, source, mask, ddq_mask=None, rolling_mask=None, force_reference=None):
    result = dict(kind=kind, t=np.round(data['t'], 6).tolist(), source=source,
                  real={}, sim={}, metrics={}, errors=dict(windowSeconds=.5, quantities={}))
    for quantity in ('q', 'dq', 'ddq', 'tau'):
        a, b = data['real_' + quantity], data['sim_' + quantity]
        valid = mask if quantity != 'ddq' or ddq_mask is None else ddq_mask
        result['real'][quantity] = np.round(a, 6).tolist()
        result['sim'][quantity] = np.round(b, 6).tolist()
        result['metrics'][quantity] = [metric(a[:, j], b[:, j], valid) for j in range(12)]
        rolling_valid = valid if rolling_mask is None else rolling_mask
        per_joint, combined = rolling_rmse(a, b, rolling_valid, samples=100)
        result['errors']['quantities'][quantity] = dict(joint=nullable(per_joint), combined=nullable(combined))
    a, b = data['sensor_counts'], data['sim_fz_N']
    reference = force_reference or dict(
        real_mean=a[mask].mean(axis=0).tolist(), real_std=a[mask].std(axis=0).tolist(),
        sim_mean=b[mask].mean(axis=0).tolist(), sim_std=b[mask].std(axis=0).tolist())
    force_shape_error = []
    for foot in range(4):
        x, y = a[mask, foot], b[mask, foot]
        if min(reference['real_std'][foot], reference['sim_std'][foot]) > 1e-10:
            x = (x - reference['real_mean'][foot]) / reference['real_std'][foot]
            y = (y - reference['sim_mean'][foot]) / reference['sim_std'][foot]
            force_shape_error.append(float(np.sqrt(np.mean((x - y) ** 2))))
        else:
            force_shape_error.append(None)
    result['force'] = dict(real_counts=np.round(a, 5).tolist(), sim_N=np.round(b, 5).tolist(),
        zscoreRmse=force_shape_error,
        correlation=[metric(a[:, j], b[:, j], mask)['correlation'] for j in range(4)], **reference)
    result['metricScope'] = 'same steady-walking samples before/after; same filter and derivative'
    result['evaluationMask'] = np.asarray(mask, bool).tolist()
    return result


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
        filter='both q/dq: fourth-order zero-phase 15 Hz LPF; ddq: derivative of filtered dq',
        resampling_dt_s=.005, simulation_executed=False)
    return real, mapped, contract


def map_pair(raw, rm, sim, sm, policy_dir=None):
    raw, sim, contract = named_mapping(raw, rm, sim, sm)
    data, timing = prepare_pair(raw, sim, 15., .005)
    sync, events = estimate_alignment(data, timing['move_start_s'], timing['move_stop_s'])
    lag = sync['applied_lag_s']
    aligned = apply_alignment(data, lag)
    # Both views use the exact same overlap, valid mask, and evaluation samples.
    query = data['t'] + lag
    keep = (query >= data['t'][0]) & (query <= data['t'][-1])
    before = {key: value.copy() for key, value in aligned.items() if not key.endswith('_unaligned')}
    shifted_valid = np.interp(query[keep], data['t'], data['valid'].astype(float)) > .999
    valid = before['valid'] & shifted_valid
    steady = before['steady'] & valid
    if steady.sum() < 100:
        raise ValueError('Insufficient common steady-walking data')
    for record in (before,):
        record['valid'], record['steady'] = valid.copy(), steady.copy()
        record['sensor_counts'] = record['real_foot_sensor_raw']
        record['sim_fz_N'] = record['sim_foot_fz_N']
    deployment = load_contract(rm['checkpoint_sha256'], policy_dir)
    after, coordinates, roundtrip = apply_contract(before, deployment, raw, sim)
    contract['time_mapping'] = sync['sign']
    contract['time_alignment_in_both_views'] = True
    contract['deployment'] = dict(**deployment['provenance'],**roundtrip)
    return before, after, timing, sync, events, contract, coordinates


def paired_paths(data_dir, dataset=DEFAULT_DATASET):
    """Select only the five requested trials from one matching dataset folder."""
    pairs = [(data_dir / 'real' / dataset / f'{name}.npz',
              data_dir / 'sim' / dataset / f'{name}.npz') for name in TRIALS]
    missing = [str(path) for pair in pairs for path in pair if not path.is_file()]
    if missing:
        raise FileNotFoundError('Required real/sim trial files are missing:\n' + '\n'.join(missing))
    return pairs


def build(data_dir, output, result_dir, policy_dir=None, dataset=DEFAULT_DATASET):
    pairs = paired_paths(data_dir, dataset)
    output.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema='go2-actor-mapping-v4', dataset=dataset, joints=list(JOINTS), feet=list(LEGS),
        units=dict(q='rad', dq='rad/s', ddq='rad/s²', tau='N·m', realForce='counts', simForce='N'), trials=[])
    results = []
    for real_path, sim_path in pairs:
        name = real_path.stem
        raw, rm = load(real_path)
        sim, sm = load(sim_path)
        if not sm['completed'] or rm['checkpoint_sha256'] != sm['checkpoint_sha256']:
            raise ValueError('Invalid simulation/reference pair')
        before, after, timing, sync, events, contract, coordinates = map_pair(raw, rm, sim, sm, policy_dir)
        first = series(before, 'before', 'Existing real/sim logs; time aligned baseline', before['steady'], rolling_mask=before['valid'])
        ref = {key: first['force'][key] for key in ('real_mean', 'real_std', 'sim_mean', 'sim_std')}
        second = series(after, 'after', 'Same time aligned logs; actual deployed actor input mapping without numeric inverse', after['steady'], rolling_mask=after['valid'], force_reference=ref)
        first['units'] = dict(q='rad',dq='rad/s',ddq='rad/s²',tau='N·m')
        second['units'] = dict(q='무차원',dq='무차원',ddq='rad/s²',tau='N·m')
        first['space'] = 'time-aligned physical measurements'
        second['space'] = 'q/dq: deployed normalized actor input; ddq/tau/force: physical (no deployment rule)'
        second['physicalMetrics'] = first['metrics']
        for item in (first, second): item['torqueSemantics'] = dict(real='ls_tau: estimated motor torque', sim='ls_tau: simulated actuator torque', mapping='joint order only; no deployment torque scale/offset')
        for item in (first, second): item['suggestedTime'] = timing['move_start_s'] + 4.5
        entry = dict(id=name, file=f'{name}.json', velocity=rm['vx'], syncPassed=sync['passed'],
            dataset=dataset, realPath=str(real_path.resolve()), simPath=str(sim_path.resolve()),
            lagMs=round(sync['applied_lag_s'] * 1000, 1), realSha256=sha(real_path), simSha256=sha(sim_path))
        trial = dict(trial=name, velocity=rm['vx'], synchronization=sync, timing=timing,
            mapping=contract, provenance=entry, before=first, after=second)
        (output / entry['file']).write_text(json.dumps(trial, separators=(',', ':'), allow_nan=False))
        manifest['trials'].append(entry)
        # Keep numeric mapping artifacts separate from historical free-replay outputs.
        arrays = dict(t=before['t'], valid=before['valid'], steady=before['steady'])
        for key in ('real_q', 'real_dq', 'real_ddq', 'real_tau'):
            arrays[key] = before[key]  # legacy physical alias
            arrays[key+'_before'] = before[key]; arrays[key+'_after'] = after[key]
        arrays['real_foot_sensor_raw'] = before['real_foot_sensor_raw']
        for key in ('sim_q', 'sim_dq', 'sim_ddq', 'sim_tau', 'sim_foot_fz_N'):
            arrays[key + '_before'] = before[key]; arrays[key + '_after'] = after[key]
        np.savez_compressed(result_dir / f'{name}_mapping.npz', **arrays, **coordinates)
        summary = dict(trial=name, provenance=entry, mapping=contract, synchronization=sync, events=events,
            n_steady=int(before['steady'].sum()), metrics={})
        for label, record, view in (('before', before, first), ('after', after, second)):
            mask = record['steady']
            summary['metrics'][label] = dict(
                **{q: metric(record['real_'+q].reshape(-1), record['sim_'+q].reshape(-1), np.repeat(mask, 12)) for q in ('q','dq','ddq','tau')},
                units=view['units'], joints=view['metrics'], force=view['force']['zscoreRmse'], force_correlation=view['force']['correlation'])
        results.append(summary)
        print(f'{name}: existing logs only; common shift {entry["lagMs"]:+.1f} ms; {summary["n_steady"]} paired evaluation samples', flush=True)
    (output / 'index.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
    (result_dir / 'mapping_summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False))
    write_report(results, result_dir)
    return manifest


def write_report(results, directory):
    import csv
    rows = []
    report = ['# 시간 정렬 기준과 배포 관측 매핑 결과', '',
        f"입력 데이터셋: `{results[0]['provenance']['dataset']}`. 대상은 `0.2_1.npz`~`0.2_5.npz`의 real/sim 쌍만 사용한다.",
        '01은 시간 정렬과 공통 전처리를 완료한 기존 물리 관측이다. 02는 같은 데이터의 q/dq에 실제 배포 정책 입력 변환을 적용해 그대로 표시한다. 숫자 역변환은 하지 않는다.',
        '수집 로그 수정, 재시뮬레이션, 정책 추론, 회귀 보정은 수행하지 않았다.',
        'q_actor = (q_sdk[order] - default - mean_q) / denominator_q',
        'dq_actor = (dq_sdk[order] - mean_dq) / denominator_dq',
        'order/default는 policy.json, mean/denominator는 실제 policy.onnx 입력 Sub/Div 상수다. ONNX 해시에 연결된 추출 파일을 사용하며 로그에서 통계를 피팅하지 않는다.',
        '그래프에서는 같은 관절 이름끼리 비교하도록 배열 인덱스만 다시 대응한다. 기본각을 더하거나 정규화를 풀지 않는다.',
        '두 도메인에 동일한 배포 변환을 적용한다. 01의 q/dq RMSE는 물리 단위, 02는 무차원이므로 전후 숫자의 감소를 물리 오차 개선률로 해석하지 않는다.',
        'ddq·tau·발 힘은 정책 관측에 없으며 배포 스케일 규약이 없어 유지한다. ddq는 rad/s², tau는 N·m, 발 센서는 counts, sim Fz는 N이다.',
        'action_scale=0.25는 정책 출력→목표각 변환에만 해당하며 측정 q/dq에 적용하지 않는다. 기록 목표각의 action 좌표는 별도로 저장한다.',
        '양쪽/전후 공통 15 Hz 필터와 dq 미분 정의, 동일 시간축·평가 마스크를 유지한다. 위상 검증 미통과 시행은 잠정 정렬이다.', '',
        '| 시행 | 시간 이동(ms) | 위상 검증 | 01 q RMSE(rad) | 02 q RMSE(무차원) | 01 dq RMSE(rad/s) | 02 dq RMSE(무차원) | ddq RMSE(rad/s², 유지) |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in results:
        b,a=r['metrics']['before'],r['metrics']['after']
        report.append(f"| {r['trial']} | {r['synchronization']['applied_lag_s']*1000:+.1f} | {'통과' if r['synchronization']['passed'] else '검토 필요'} | {b['q']['rmse']:.4f} | {a['q']['rmse']:.4f} | {b['dq']['rmse']:.4f} | {a['dq']['rmse']:.4f} | {a['ddq']['rmse']:.4f} |")
        for mode in ('before','after'):
            for q in ('q','dq','ddq','tau'):
                for j,m in zip(JOINTS,r['metrics'][mode]['joints'][q]):
                    rows.append(dict(trial=r['trial'], mode=mode, channel=j, quantity=q, unit=r['metrics'][mode]['units'][q], **m))
            for f,rmse,corr in zip(LEGS,r['metrics'][mode]['force'],r['metrics'][mode]['force_correlation']):
                rows.append(dict(trial=r['trial'], mode=mode, channel=f, quantity='force_zscore',unit='dimensionless',rmse=rmse,correlation=corr,n=r['n_steady']))
    report += ['', '## 토크 비교 (N·m)', '', 'real ls_tau는 모터 추정 토크, sim ls_tau는 액추에이터 토크다. 양쪽에 같은 15 Hz 필터와 기존 공통 시간 이동을 적용했다. 배포 파일에 토크 배율/오프셋이 없어 02는 관절 이름 대응만 적용하며 수치는 유지한다.', '', '| 시행 | 01 tau RMSE | 02 tau RMSE |', '|---|---:|---:|']
    report += [f"| {r['trial']} | {r['metrics']['before']['tau']['rmse']:.4f} | {r['metrics']['after']['tau']['rmse']:.4f} |" for r in results]
    report += ['', '## 발 힘 파형 오차', '', '| 시행 | FR z-RMSE 전→후 | FL | RR | RL |', '|---|---:|---:|---:|---:|']
    for r in results:
        report.append('| '+r['trial']+' | '+' | '.join('—' if b is None or a is None else f'{b:.3f} → {a:.3f}' for b,a in zip(r['metrics']['before']['force'],r['metrics']['after']['force']))+' |')
    report += ['', '## 배포 규약 검증', '',
        'policy.onnx 해시와 checkpoint 식별자를 확인하고 samples.json/parity.npz의 관절 관측 변환을 검증했다. 정책 추론은 실행하지 않았다.',
        'ONNX 내부 정규화를 02의 q/dq에 한 번 적용한다. ddq와 발 힘에는 적용하지 않는다.',
        '실제/시뮬 목표각에서 역산한 action 좌표는 기록된 목표각의 표현이며 원래 정책 raw action의 완전한 복원을 의미하지 않는다.', '',
        '| 시행 | q 왕복 최대오차(rad) | 목표각 왕복 최대오차(rad) |', '|---|---:|---:|']
    for r in results:
        d=r['mapping']['deployment']
        report.append(f"| {r['trial']} | {max(d['real_q_roundtrip_max_error_rad'],d['sim_q_roundtrip_max_error_rad']):.3e} | {max(d['real_target_roundtrip_max_error_rad'],d['sim_target_roundtrip_max_error_rad']):.3e} |")
    biases = [(r['trial'], max(zip(JOINTS, r['metrics']['after']['joints']['q']), key=lambda item: abs(item[1]['bias']))) for r in results]
    report += ['', '## 02 정책 입력 공간의 관절 평균 차이', '', '| 시행 | 평균 각도 차이가 가장 큰 관절 | sim − real 평균(무차원) |', '|---|---|---:|']
    report += [f"| {trial} | {joint} | {value['bias']:+.4f} |" for trial,(joint,value) in biases]
    report += ['', '정책 공간의 평균 차이는 고정 정규화 배율로 가중된 차이다. 물리 단위의 잔여 오차는 01 수치를 참조한다.']
    report += ['', '## 남는 차이의 해석', '',
        '02의 q/dq는 좌표 표현이 바뀌므로 01과 값 및 RMSE가 다를 수 있다. 배포 좌표 변환은 시스템 물리 오차를 교정하지 않는다. ddq·발 힘의 전후 동일성은 규약이 없는 항목을 유지한 결과다.',
        '보정 후 남는 관절 RMSE에는 보행 진폭·파형·평균 자세 차이와 정렬 잔차가 포함된다. 특정 모터·질량·마찰 오차로 단정할 수 없다.',
        '위상 검증이 통과하지 않은 시행은 잠정 정렬이며 후반 위상 드리프트 가능성이 남는다. 시작에 사용하지 않은 끝 구간으로 검증했다.',
        '발 센서와 sim world Fz의 측정 정의·단위가 다르므로 절대 힘 오차는 산출하지 않았다. 이 비교만으로 GRF predictor의 N 단위 정확도를 검증할 수 없다.',
        '발 힘은 배포 정책의 입력에 포함되지 않으며 counts→N 교정 계수도 없다. 따라서 01/02의 발 힘과 파형 RMSE는 동일하다.',
        '기존 figure_after의 LowCmd 재생 산출물은 과거 별도 실험이며 이번 UI와 분석에서 읽지 않는다.', '',
        '## 그림', '', '[전체 시행 RMSE 요약](figure/mapping_rmse_summary.png)', '']
    report += [f"- [{r['trial']} 관절 배포 입력 매핑](figure/{r['trial']}_joint_mapping.png) · [발 힘(유지)](figure/{r['trial']}_foot_force_before_after.png)" for r in results]
    (directory/'analysis.md').write_text('\n'.join(report)+'\n')
    with (directory/'metrics.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['trial','mode','channel','quantity','unit','n','rmse','mae','bias','correlation'])
        writer.writeheader(); writer.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=PROJECT / 'data')
    parser.add_argument('--dataset', default=DEFAULT_DATASET,
                        help='Matching subfolder under data/real and data/sim (default: %(default)s)')
    parser.add_argument('--output-dir', type=Path, default=BASE / 'web/data')
    parser.add_argument('--result-dir', type=Path, default=PROJECT / 'result/data_mapping')
    parser.add_argument('--policy-dir', type=Path, default=None)
    args = parser.parse_args()
    build(args.data_dir, args.output_dir, args.result_dir, args.policy_dir, args.dataset)
