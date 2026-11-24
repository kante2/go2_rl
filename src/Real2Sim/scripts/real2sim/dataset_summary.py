"""Physical real/sim TXT report from a completed, saved result bundle."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'real2sim'

from .config import DEFAULT_DATASET, JOINTS, LEGS, result_path


QUANTITIES = ('q', 'dq', 'tau')
UNITS = dict(q='rad', dq='rad/s', tau='N·m')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _number(value, digits=4):
    return 'NA' if value is None or not np.isfinite(value) else f'{value:.{digits}f}'


def _physical(rows, quantity, mask='steady', clock=False, columns=None):
    columns = list(range(12)) if columns is None else columns
    real, sim, correlations, weights, mean_errors = [], [], [], [], []
    for row in rows:
        data = row['data']; selected = data[mask]
        a = data['real_' + quantity + '_before'][selected][:, columns]
        b = data['sim_' + quantity + ('_clock_only' if clock else '_before')][selected][:, columns]
        real.append(a.ravel()); sim.append(b.ravel())
        for j in range(len(columns)):
            if min(a[:, j].std(), b[:, j].std()) > 1e-10:
                correlations.append(np.corrcoef(a[:, j], b[:, j])[0, 1]); weights.append(len(a))
        mean_errors.extend([(float(x), len(a)) for x in (b-a).mean(0)])
    if not rows:
        return dict.fromkeys(('n', 'rmse', 'mae', 'bias', 'p95', 'r', 'bias_rms', 'centered_rmse'))
    error = np.concatenate(sim) - np.concatenate(real)
    mse = float(np.mean(error ** 2))
    bias_mse = float(np.average([x*x for x, _ in mean_errors], weights=[n for _, n in mean_errors]))
    return dict(n=len(error), rmse=np.sqrt(mse), mae=np.mean(abs(error)), bias=np.mean(error),
        p95=np.percentile(abs(error), 95), r=np.average(correlations, weights=weights) if weights else None,
        bias_rms=np.sqrt(bias_mse), centered_rmse=np.sqrt(max(0., mse-bias_mse)))


def _force(rows, mask='steady', clock=False, foot=None):
    columns = list(range(4)) if foot is None else [foot]
    errors, correlations, weights, real, sim = [], [], [], [], []
    for row in rows:
        data = row['data']; selected = data[mask]; reference = row['detail']['before']['force']
        a = data['real_foot_sensor_raw'][selected][:, columns]
        b = data['sim_foot_fz_N_clock_only' if clock else 'sim_foot_fz_N_before'][selected][:, columns]
        real.append(a.ravel()); sim.append(b.ravel())
        az = (a - np.asarray(reference['real_mean'])[columns]) / np.asarray(reference['real_std'])[columns]
        bz = (b - np.asarray(reference['sim_mean'])[columns]) / np.asarray(reference['sim_std'])[columns]
        errors.append((bz-az).ravel())
        for j in range(len(columns)):
            if min(a[:, j].std(), b[:, j].std()) > 1e-10:
                correlations.append(np.corrcoef(a[:, j], b[:, j])[0, 1]); weights.append(len(a))
    if not rows:
        return dict.fromkeys(('zrmse', 'r', 'real_mean', 'sim_mean', 'real_p95', 'sim_p95'))
    return dict(zrmse=np.sqrt(np.mean(np.concatenate(errors)**2)),
        r=np.average(correlations, weights=weights) if weights else None,
        real_mean=np.mean(np.concatenate(real)), sim_mean=np.mean(np.concatenate(sim)),
        real_p95=np.percentile(np.concatenate(real), 95), sim_p95=np.percentile(np.concatenate(sim), 95))


def _contact_rows(rows):
    # Reuse the shared contact detector; no additional force-only shift.
    from diagnose_contact_sync import contact_events, match_onsets
    for row in rows:
        data = row['data']; t = data['t']; sync = row['detail']['synchronization']
        calibration = sync['calibration']['window_s']
        _, re, real_states = contact_events(t, data['real_foot_sensor_raw'], data['valid'], calibration)
        _, se, sim_states = contact_events(t, data['sim_foot_fz_N_before'], data['valid'], calibration)
        row['contacts'] = {}
        for mask_name in ('steady', 'late'):
            mask = data[mask_name]; row['contacts'][mask_name] = []
            for j in range(4):
                events = [[v for v in stamps if np.interp(v, t, mask.astype(float), left=0, right=0) > .999]
                          for stamps in (re[j], se[j])]
                matched = match_onsets(events[0], events[1], [float(t[0]), float(t[-1])])
                good = mask & np.isfinite(real_states[:, j]) & np.isfinite(sim_states[:, j])
                row['contacts'][mask_name].append(dict(n=int(good.sum()),
                    disagreement=int(np.count_nonzero(real_states[good, j] != sim_states[good, j])),
                    real_on=int(np.count_nonzero(real_states[good, j] == 1)),
                    sim_on=int(np.count_nonzero(sim_states[good, j] == 1)),
                    real_events=len(events[0]), sim_events=len(events[1]),
                    residuals=[item['residual_ms'] for item in matched['pairs']]))


def _contacts(rows, mask='steady', foot=None):
    items = [record for row in rows for j, record in enumerate(row['contacts'][mask]) if foot is None or j == foot]
    counts = {key: sum(record[key] for record in items) for key in ('n', 'disagreement', 'real_on', 'sim_on', 'real_events', 'sim_events')}
    residuals = [v for record in items for v in record['residuals']]
    return dict(**counts, matched=len(residuals), median=np.median(residuals) if residuals else None,
        p90=np.percentile(abs(np.asarray(residuals)), 90) if residuals else None,
        coverage=100*len(residuals)/counts['real_events'] if counts['real_events'] else None,
        disagreement_pct=100*counts['disagreement']/counts['n'] if counts['n'] else None,
        real_duty=100*counts['real_on']/counts['n'] if counts['n'] else None,
        sim_duty=100*counts['sim_on']/counts['n'] if counts['n'] else None)


def write_summary(directory, output, display_directory=None):
    """Read saved physical arrays; verify sources; atomically write one TXT."""
    directory, output = Path(directory), Path(output)
    shown = Path(display_directory) if display_directory is not None else directory
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('status') != 'complete':
        raise ValueError('TXT summary requires a completed result')
    rows, hashes, raw_counts = [], [], {}
    entries = manifest['trials'] + manifest['excludedTrials']
    for entry in entries:
        for domain in ('real', 'sim'):
            path = Path(entry[domain + 'Path']); actual = _sha(path)
            if entry.get(domain + 'Sha256', actual) != actual:
                raise ValueError(f'Source changed: {path}')
            hashes.append((path, actual))
            with np.load(path, allow_pickle=False) as z:
                raw_counts[entry['id'], domain] = len(z['ls_t'])
    for domain in ('real', 'sim'):
        expected = {Path(entry[domain + 'Path']).resolve() for entry in entries}
        parents = {path.parent for path in expected}
        if {path.resolve() for parent in parents for path in parent.glob('*.npz')} != expected:
            raise ValueError('Recording inventory changed; rebuild before writing the TXT')
    for entry in manifest['trials']:
        name = entry['id']; metrics_path = directory / 'metrics' / f'{name}.json'
        if _sha(metrics_path) != manifest['artifacts'][f'metrics/{name}.json']:
            raise ValueError('Saved metrics changed')
        detail = json.loads(metrics_path.read_text())
        relative = f'processed/{name}_mapping.npz'
        if _sha(directory / relative) != manifest['artifacts'][relative]:
            raise ValueError('Saved processed data changed')
        keys = ['t', 'valid', 'steady', 'real_foot_sensor_raw', 'sim_foot_fz_N_before', 'sim_foot_fz_N_clock_only']
        keys += [f'{domain}_{q}_{mode}' for q in QUANTITIES for domain in ('real', 'sim')
                 for mode in (('before', 'clock_only') if domain == 'sim' else ('before',))]
        with np.load(directory / relative, allow_pickle=False) as z:
            data = {key: z[key] for key in keys}
        left, right = detail['synchronization']['validation']['window_s']
        data['late'] = data['steady'] & (data['t'] >= left) & (data['t'] <= right)
        row = dict(name=name, detail=detail, data=data, accepted=detail['synchronization']['passed'],
            group=name.rsplit('_v', 1)[0] if '_v' in name else name.rsplit('_', 1)[0])
        for q in QUANTITIES:
            error = data['sim_'+q+'_before'][data['steady']] - data['real_'+q+'_before'][data['steady']]
            np.testing.assert_allclose(np.sqrt(np.mean(error**2, axis=0)),
                [record['rmse'] for record in detail['before']['metrics'][q]], atol=1e-12)
        for j in range(4):
            np.testing.assert_allclose(_force([row], foot=j)['zrmse'], detail['before']['force']['zscoreRmse'][j], atol=1e-12)
        rows.append(row)
    if not rows:
        raise ValueError('No comparable recordings')
    _contact_rows(rows)
    accepted = [row for row in rows if row['accepted']]
    rejected = [row for row in rows if not row['accepted']]
    cohorts = [('전체', rows)] + ([('정렬 통과', accepted)] if rejected and accepted else []) + ([('정렬 미통과', rejected)] if rejected else [])
    dt = manifest['settings']['dt_s']; frames = sum(int(row['data']['steady'].sum()) for row in rows)
    lines = []
    def add(*text): lines.extend(str(item) for item in text)
    def section(title): add('', '='*105, title, '='*105)
    def table(headers, values):
        add(' | '.join(headers), '-'*105)
        for values_row in values: add(' | '.join(str(value) for value in values_row))
    add(f'{manifest["dataset"]}: 실제 로봇 / 시뮬레이터 접촉력·q·dq·tau 차이 분석',
        f'작성 시각(KST): {datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")}',
        f'분석 결과 생성 시각(UTC): {manifest["generatedAt"]}',
        f'전체 입력 {len(entries)}쌍, 비교 가능 {len(rows)}쌍, 빈 상태 기록 제외 {len(manifest["excludedTrials"])}쌍.',
        f'주 평가 {frames:,}프레임, 등가 표본시간 {frames*dt:.3f}초. 관절량당 {frames*12:,}점, 발-시각 {frames*4:,}점.',
        f'시간 정렬 검증 통과 {len(accepted)}쌍 / 미통과 {len(rejected)}쌍. 현재 데이터 파일명과 원본 SHA256 기준으로 작성했다.')
    section('1. 핵심 수치: 물리 단위의 관측 오차')
    table(['범위','q RMSE[rad / deg]','dq RMSE[rad/s]','tau RMSE[N·m]','접촉력 z-RMSE','접촉력 r_mean'],[
        [f'{label} {len(group)}쌍',f'{_physical(group,"q")["rmse"]:.4f} / {np.degrees(_physical(group,"q")["rmse"]):.3f}',
         *[_number(_physical(group,q)['rmse']) for q in ('dq','tau')],_number(_force(group)['zrmse']),_number(_force(group)['r'])] for label,group in cohorts])
    add('접촉력은 real 센서 counts와 sim 수직하중 N을 비교한다. 절대 힘 RMSE[N]와 오차율[%]는 센서 교정 없이 계산할 수 없다.',
        'z-RMSE와 r은 파형 일치도다. q/dq/tau는 배포 정규화 전 물리값을 사용했다.')
    section('2. 전처리, 시간 정렬 및 집계 정의')
    add('관절 순서: FR, FL, RR, RL; 각 다리 hip, thigh, calf. 오차 부호는 sim-real.',
        'real 약 500Hz 상태는 tick 중복을 제거하고 로봇 시간에 대응시킨다. sim 약 200Hz 상태는 sim_time에서 처리한다.',
        'q/dq/tau에 양쪽 같은 4차 zero-phase 15Hz 저역통과 필터를 적용하고 5ms 공통 시간축으로 보간한다.',
        'steady 평가: 녹음 시작 2초·끝 1초, 필터 경계, 누락 주변, 정렬 후 겹치지 않는 부분을 제외한다. 정상 보행이 자동 검증된 구간은 아니다.',
        '실제 tau는 모터 추정 토크, sim tau는 액추에이터 토크다. 토크 센서 정답 대비 정확도 검증은 아니다.',
        'real 발 센서는 ls_ff의 counts, sim은 -sim_foot_force_vec[:,:,2]의 양의 수직하중 N이다. sim ls_ff의 공중 오프셋은 쓰지 않는다.',
        '접촉력은 시간 보간한 값을 사용하며 관절량의 15Hz 필터를 추가 적용하지 않는다.',
        '기본 ±250ms 탐색은 앞 구간 q/dq로 추정하고 뒤 구간으로 검증한다. 경계에 도달한 경우에만 ±500ms로 한 번 확장한다.',
        '확장 시 앞 구간의 유일한 peak(대안 대비 상관 차이 0.02 이상)가 주기를 선택한다. 뒤 구간은 그 주기 내 ±80ms에서 잔차를 검증한다.',
        '뒤 구간의 고정 후보 상관≥0.70, 앞뒤 후보차≤20ms, 더 좋은 다른 후보와의 상관차≤0.02를 요구한다. 앞 구간의 주기 모호함과 시간 변화는 거부한다.',
        '뒤 구간에는 보행 한 주기 차이의 동점이 남을 수 있다. 앞에서 선택한 주기를 유지하는 파형 정렬이며 순수 통신 지연 추정값이 아니다.',
        '모든 sim q/dq/ddq/tau/발 힘에 하나의 이동을 적용한다. 발·관절별 추가 이동, 진폭 피팅, 재시뮬레이션은 없다.',
        'lag>0: aligned_sim(t)=original_sim(t+lag), 즉 sim 파형을 왼쪽으로 이동한다. lag<0은 오른쪽으로 이동한다.',
        '접촉 기준 예외는 네 발 하중의 강한 상관과 관절 후보의 일치를 앞뒤 구간 모두에서 요구한다.',
        'RMSE=sqrt(mean(e²)), MAE=mean(abs(e)), bias=mean(e), P95=절대오차의 95백분위.',
        '전체 RMSE는 시각·관절의 제곱오차를 표본수 가중해 합친 값이다. 시행 균등 RMSE 평균도 별도로 제시한다.',
        'r_mean은 시행·관절(발)의 Pearson r을 프레임 수로 가중한 평균이다. 관절 기본각을 펼친 단일 상관계수가 아니다.',
        '접촉력 z-score는 각 시행·발·도메인의 steady 평균/표준편차로 계산한다. 시간 이동 전후에도 이동 후의 같은 통계를 사용한다.',
        '뒤 검증구간은 후보 추정에 쓰지 않지만 수용 판정에 쓰인다. 독립 테스트셋으로 주장하지 않는다. 시각·관절 표본은 서로 독립이 아니다.',
        '수집기에 당시 정책 해시·전체 정책 입력이 없어 모델 신원과 입력 47차원 전체를 로그만으로 복원할 수 없다.')
    section('3. 전체 오차, 시간 이동 효과, 평가 구간 민감도')
    table(['량','RMSE','MAE','bias','절대오차 P95','r_mean','단위'],[
        [q,*[_number(_physical(rows,q)[key]) for key in ('rmse','mae','bias','p95','r')],UNITS[q]] for q in QUANTITIES])
    table(['량','PC 시각만 RMSE','공통 이동 후 RMSE','감소율[%]','시행 RMSE 균등평균','단위'],[
        [q,_number(_physical(rows,q,clock=True)['rmse']),_number(_physical(rows,q)['rmse']),
         _number(100*(1-_physical(rows,q)['rmse']/_physical(rows,q,clock=True)['rmse']),2),
         _number(np.mean([_physical([row],q)['rmse'] for row in rows])),UNITS[q]] for q in QUANTITIES])
    add('감소율은 같은 물리값과 평가 마스크에서 시간 정렬로 발생한 변화다. 동역학 모델을 개선한 결과가 아니다.')
    table(['구간','프레임 수','q RMSE[rad]','dq RMSE[rad/s]','tau RMSE[N·m]','접촉력 z-RMSE','접촉력 r_mean'],[
        [mask,sum(int(row['data'][mask].sum()) for row in rows),*[_number(_physical(rows,q,mask=mask)['rmse']) for q in QUANTITIES],
         _number(_force(rows,mask=mask)['zrmse']),_number(_force(rows,mask=mask)['r'])] for mask in ('steady','late','valid')])
    table(['파일명 그룹','시행 수','q RMSE[rad]','dq RMSE[rad/s]','tau RMSE[N·m]','접촉력 r_mean'],[
        [name,len(group),*[_number(_physical(group,q)['rmse']) for q in QUANTITIES],_number(_force(group)['r'])]
        for name in dict.fromkeys(row['group'] for row in rows) for group in [[row for row in rows if row['group']==name]]])
    add('파일명의 숫자는 이름에 따른 그룹이다. 실제 기록된 속도 명령으로 단정하지 않았다.')
    section('4. 모든 시행의 결과와 시간 이동')
    table(['시행','real/sim 원본 n','평가 n','통과','lag[ms]','q RMSE[rad]','dq RMSE[rad/s]','tau RMSE[N·m]','발 z-RMSE','발 r_mean'],[
        [row['name'],f'{raw_counts[row["name"],"real"]}/{raw_counts[row["name"],"sim"]}',int(row['data']['steady'].sum()),row['accepted'],
         f'{row["detail"]["synchronization"]["applied_lag_s"]*1000:+.0f}',*[_number(_physical([row],q)['rmse']) for q in QUANTITIES],
         _number(_force([row])['zrmse']),_number(_force([row])['r'])] for row in rows])
    table(['시행','q MAE / bias / P95[rad]','dq MAE / bias / P95[rad/s]','tau MAE / bias / P95[N·m]'],[
        [row['name'],*[' / '.join(_number(_physical([row],q)[key]) for key in ('mae','bias','p95')) for q in QUANTITIES]] for row in rows])
    for row in rows:
        sync = row['detail']['synchronization']
        if sync.get('search_expansion'):
            add(f'{row["name"]}: ±500ms 확장. 앞 상관 {sync["calibration"]["correlation_best"]:.4f}, 대안 대비 차이 {sync["calibration"]["peak_margin"]:.4f}; '
                f'뒤 고정 후보 상관 {sync["validation"]["correlation_at_calibration_lag"]:.4f}, 같은 주기 내 잔차 {sync["held_out_lag_difference_s"]*1000:+.1f}ms. '
                f'뒤 전역 주기 모호={sync["validation"]["ambiguous"]}.')
        if not row['accepted']: add(f'{row["name"]} 미적용 사유: {", ".join(sync["rejection_reasons"])}')
    for entry in manifest['excludedTrials']: add(f'제외 {entry["id"]}: {entry["reason"]}')
    section('5. 12관절별 물리 오차')
    for label, group in cohorts:
        for q in QUANTITIES:
            add('',f'{label} / {q} [{UNITS[q]}]')
            table(['관절','RMSE','MAE','bias','절대오차 P95','r_mean'],[
                [name.removesuffix('_joint'),*[_number(_physical(group,q,columns=[j])[key]) for key in ('rmse','mae','bias','p95','r')]] for j,name in enumerate(JOINTS)])
    add('작은 운동 범위의 hip은 작은 절대 오차에도 상관계수가 낮을 수 있다. signed bias의 전체 평균에는 좌우 상쇄가 포함된다.')
    section('6. 접촉력: 발별 파형과 원래 측정 크기')
    table(['발','z-RMSE','r_mean','real 평균[counts]','real P95[counts]','sim 평균[N]','sim P95[N]'],[
        [foot,*[_number(_force(rows,foot=j)[key]) for key in ('zrmse','r','real_mean','real_p95','sim_mean','sim_p95')]] for j,foot in enumerate(LEGS)])
    table(['시행',*['%s z-RMSE / r'%foot for foot in LEGS]],[
        [row['name'],*[f'{_number(_force([row],foot=j)["zrmse"])} / {_number(_force([row],foot=j)["r"])}' for j in range(4)]] for row in rows])
    add('각 도메인의 평균·진폭을 제거한 지표다. 같은 steady 구간에서 z-RMSE²=2(1-r). 절대 힘이나 충격 크기가 일치한다는 뜻은 아니다.',
        '접촉력의 절대 오차를 얻으려면 real 센서의 영점·counts→N·측정축과 sim 수직성분의 정의를 먼저 맞춰야 한다.')
    section('7. 착지 시각 및 접촉 상태 비교')
    add('각 시행·발·도메인 앞 구간 p10/p90으로 on=p10+15%*(p90-p10), off=p10+7%*(p90-p10)를 설정했다.',
        'hysteresis, 6ms 지속 확인, 최소 40ms 이륙 후 착지, 같은 발끼리 ±150ms 이내 1:1 근접 매칭. 해상도는 5ms다.',
        '양수 잔차는 sim 문턱 기반 착지가 real보다 늦다는 뜻이다. 센서 응답 및 파형 차이가 포함되며 하드웨어 지연이 아니다.',
        '미매칭 사건의 잔차는 0으로 대체하지 않는다. 상태 불일치율과 duty는 두 검출 상태가 모두 유효한 시각에서 계산한다.')
    table(['범위/구간','매칭/real착지','sim착지','매칭률[%]','잔차 중앙값[ms]','절대잔차 P90[ms]','상태 불일치[%]','real duty[%]','sim duty[%]'],[
        [label+'/'+mask,f'{c["matched"]}/{c["real_events"]}',c['sim_events'],*[_number(c[key],2) for key in ('coverage','median','p90','disagreement_pct','real_duty','sim_duty')]]
        for label,group in cohorts for mask in ('steady','late') for c in [_contacts(group,mask=mask)]])
    table(['발','매칭/real착지','잔차 중앙값[ms]','절대잔차 P90[ms]','상태 불일치[%]'],[
        [foot,f'{c["matched"]}/{c["real_events"]}',*[_number(c[key],2) for key in ('median','p90','disagreement_pct')]]
        for j,foot in enumerate(LEGS) for c in [_contacts(rows,foot=j)]])
    table(['시행',*['%s 뒤 잔차중앙값[ms] / 매칭수(real수)'%foot for foot in LEGS]],[
        [row['name'],*[f'{_number(c["median"],1)} / {c["matched"]}({c["real_events"]})' for c in [_contacts([row],mask='late',foot=j) for j in range(4)]]] for row in rows])
    add('매칭률이 낮은 집단은 매칭된 잔차만으로 전체 동기를 판단할 수 없다. 보행 한 주기 차이의 실제 지연은 이 파형 비교만으로 식별되지 않는다.')
    section('8. 해석과 다음 확인')
    for q in QUANTITIES:
        worst = sorted(rows,key=lambda row:_physical([row],q)['rmse'],reverse=True)[:3]
        add(f'{q} 시행 RMSE 상위 [{UNITS[q]}]: '+ '; '.join(f'{row["name"]}={_physical([row],q)["rmse"]:.4f}' for row in worst))
        metrics = _physical(rows,q)
        add(f'  시행·관절 평균차 RMS={metrics["bias_rms"]:.4f}, 평균차 제거 잔여 RMSE={metrics["centered_rmse"]:.4f}. 평균차 제거는 진단용이며 실제 보정에 적용하지 않았다.')
    add('잔여 오차에는 평균 자세, 보행 진폭·파형, 구동기·센서 응답, 정렬 잔차가 섞여 있다. 특정 질량·마찰·게인 오류를 식별한 분석은 아니다.',
        'real 목표각 변화율 제한과 sim 제한 부재는 기록 설명의 차이 요인이다. 그 인과적 기여도는 별도 실험이 필요하다.',
        '500/200Hz는 상태·물리 기록 주기이고 현재 정책 추론은 양쪽 50Hz다. 분석용 200Hz 보간이 배포 주기를 바꾸지 않는다.',
        '다음 수집에는 정책 해시, 추론 시각, 원본 관측 47차원, raw action, 명령, 보행 위상을 남겨 위상·전달·구동기 차이를 구분해야 한다.')
    section('9. 출처, 검증 및 재현')
    add(f'결과 경로: {shown}',f'manifest SHA256: {_sha(directory / "manifest.json")}',
        f'분석 소스 합성 SHA256: {manifest["analysisCodeSha256"]}',
        'processed의 물리 q/dq/tau와 접촉력 통계를 저장된 지표와 대조했다. 원본 해시와 현재 파일 목록을 확인했다.',
        'TXT 생성: python3 -m real2sim.dataset_summary --result-dir <결과경로> --output <TXT경로> (scripts를 PYTHONPATH로 지정)',
        '',f'원본 {len(hashes)}개 SHA256:')
    for path, sha in hashes:
        if _sha(path) != sha: raise ValueError(f'Source changed during TXT generation: {path}')
        add(f'{path.parent.parent.name}/{path.name}  {sha}')
    text = '\n'.join(lines) + '\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name('.' + output.name + '.tmp')
    try:
        temporary.write_text(text, encoding='utf-8')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default=DEFAULT_DATASET)
    parser.add_argument('--result-dir', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    directory = args.result_dir or result_path(args.dataset)
    print(write_summary(directory, args.output or directory.parent / f'{args.dataset}_real_sim_analysis.txt'))
