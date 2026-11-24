"""Human-readable analysis reports from computed metrics."""
import numpy as np
from .config import JOINTS, LEGS


def write_report(results, directory, excluded=()):
    import csv
    rows = []
    report = ['# 시간 정렬 기준과 배포 관측 매핑 결과', '',
        f"입력 데이터셋: `{results[0]['provenance']['dataset']}`. 해당 폴더의 real/sim 파일 {len(results)}쌍만 사용한다. 원본 경로와 SHA256은 manifest.json에 기록한다.",
        '공통 PC 시각 위에서 앞 구간 q/dq로 공통 이동을 추정하고 뒤 구간에서 검증한다. 관절 상관만 부족한 경우에는 네 발의 높은 하중 상관과 관절 추정 시각의 일치가 독립된 뒤 구간에서도 확인될 때만 접촉 기준 이동을 적용한다. 나머지 미통과는 0ms를 유지한다. 모든 sim 채널에 같은 이동을 적용하며 sim 필터/미분은 sim_time을 사용한다.',
        '이 이동은 파형 위상 정렬이며 하드웨어 통신 지연 측정값이 아니다. 관절별 이동·진폭 피팅·시간축 늘이기는 없다.',
        '기본 ±250ms 탐색이 경계에 도달한 경우에만 ±500ms로 한 번 확장한다. 앞 구간에서 유일한 주기 후보를 선택하고 뒤 구간에서 고정 후보의 상관과 같은 주기 안의 국소 잔차를 검증한다. 뒤 구간의 한 보행 주기 차이 동점은 앞 구간의 주기 선택을 유지하며, 앞 구간부터 주기 구분이 안 되거나 시간 차이가 변하면 적용하지 않는다.',
        'move_start/stop은 녹음 구간 표시다. 평가 마스크는 시작 2초·끝 1초·데이터 누락 주변을 제외한 녹음 내부이며 정상 보행을 자동 검증한 구간이 아니다.',
        '새 수집 NPZ에는 정책 해시가 없다. data/explain_data.txt에 명시된 baseline 배포 파일 자체의 해시/parity를 검증하며 녹음 당시 정책 해시가 검증됐다고 주장하지 않는다.',
        '01은 시간 정렬과 공통 전처리를 완료한 기존 물리 관측이다. 02는 같은 데이터의 q/dq에 실제 배포 정책 입력 변환을 적용해 그대로 표시한다. 숫자 역변환은 하지 않는다.',
        '수집 로그 수정, 재시뮬레이션, 정책 추론, 회귀 보정은 수행하지 않았다.',
        'q_actor = (q_sdk[order] - default - mean_q) / denominator_q',
        'dq_actor = (dq_sdk[order] - mean_dq) / denominator_dq',
        'order/default는 policy.json, mean/denominator는 실제 policy.onnx 입력 Sub/Div 상수다. ONNX 해시에 연결된 추출 파일을 사용하며 로그에서 통계를 피팅하지 않는다.',
        '그래프에서는 같은 관절 이름끼리 비교하도록 배열 인덱스만 다시 대응한다. 기본각을 더하거나 정규화를 풀지 않는다.',
        '두 도메인에 동일한 배포 변환을 적용한다. 01의 q/dq RMSE는 물리 단위, 02는 무차원이므로 전후 숫자의 감소를 물리 오차 개선률로 해석하지 않는다.',
        'ddq·tau·발 힘은 정책 관측에 없으며 배포 스케일 규약이 없어 유지한다. ddq는 rad/s², tau는 N·m, 발 센서는 counts, sim Fz는 N이다.',
        'action_scale=0.25는 정책 출력→목표각 변환에만 해당하며 측정 q/dq에 적용하지 않는다. 기록 목표각의 action 좌표는 별도로 저장한다.',
        '양쪽/전후 공통 15 Hz 필터와 dq 미분 정의, 동일 시간축·평가 마스크를 유지한다.', '',
        '| 시행 | 시간 이동(ms) | 시간 기준 | 01 q RMSE(rad) | 02 q RMSE(무차원) | 01 dq RMSE(rad/s) | 02 dq RMSE(무차원) | ddq RMSE(rad/s², 유지) |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in results:
        b,a=r['metrics']['before'],r['metrics']['after']
        clock = '검증 통과' if r['synchronization']['passed'] else '미적용 · 검토 필요'
        report.append(f"| {r['trial']} | {r['synchronization']['applied_lag_s']*1000:+.1f} | {clock} | {b['q']['rmse']:.4f} | {a['q']['rmse']:.4f} | {b['dq']['rmse']:.4f} | {a['dq']['rmse']:.4f} | {a['ddq']['rmse']:.4f} |")
        for mode in ('before','after'):
            for q in ('q','dq','ddq','tau'):
                for j,m in zip(JOINTS,r['metrics'][mode]['joints'][q]):
                    rows.append(dict(trial=r['trial'], mode=mode, channel=j, quantity=q, unit=r['metrics'][mode]['units'][q], **m))
            for f,rmse,corr in zip(LEGS,r['metrics'][mode]['force'],r['metrics'][mode]['force_correlation']):
                rows.append(dict(trial=r['trial'], mode=mode, channel=f, quantity='force_zscore',unit='dimensionless',rmse=rmse,correlation=corr,n=r['n_steady']))
    report += ['', '## 시간 이동 자체의 영향 (물리 단위)', '',
        '01/02의 배포 좌표 변환과 별개다. 시간 이동 전후를 같은 시간축·평가 샘플·물리 단위로 비교한다. 이동 후 토크 오차가 커지는 경우도 그대로 기록한다.', '',
        '| 시행 | 시간 이동(ms) | 검증 | q RMSE 이동 전→후(rad) | dq(rad/s) | tau(N·m) |', '|---|---:|---|---|---|---|']
    for r in results:
        sync = r['synchronization']; quantities = sync['physicalComparison']['quantities']
        cells = []
        for q in ('q', 'dq', 'tau'):
            values = [np.sqrt(np.mean([m['rmse'] ** 2 for m in quantities[q][mode]])) for mode in ('clockOnly', 'aligned')]
            cells.append(f'{values[0]:.4f} → {values[1]:.4f}')
        report.append(f"| {r['trial']} | {sync['applied_lag_s']*1000:+.1f} | {'통과' if sync['passed'] else ', '.join(sync.get('rejection_reasons', []))} | " + ' | '.join(cells) + ' |')
    report += ['', '## 토크 비교 (N·m)', '', 'real ls_tau는 모터 추정 토크, sim ls_tau는 액추에이터 토크다. 양쪽에 같은 15 Hz 필터와 검증된 공통 시간 이동을 적용했다. 배포 파일에 토크 배율/오프셋이 없어 02는 관절 이름 대응만 적용하며 수치는 유지한다.', '', '| 시행 | 01 tau RMSE | 02 tau RMSE |', '|---|---:|---:|']
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
        '시간 이동은 앞 구간에서만 추정하고 뒤 구간으로 검증했다. 검증 미통과 시행은 시간 이동 없이 공통 PC 시각만 사용한다. 정책 위상·동역학 차이와 시간 변화는 남을 수 있다.',
        '발 센서와 sim world Fz의 측정 정의·단위가 다르므로 절대 힘 오차는 산출하지 않았다. 이 비교만으로 GRF predictor의 N 단위 정확도를 검증할 수 없다.',
        '발 힘은 배포 정책의 입력에 포함되지 않으며 counts→N 교정 계수도 없다. 따라서 01/02의 발 힘과 파형 RMSE는 동일하다.',
        'figures/torque의 토크 결과도 같은 입력 매핑 NPZ로 생성하며 01/02에서 동일한 토크 그림은 한 번만 저장한다.', '',
        '## 그림', '', '[전체 시행 RMSE 요약](figures/mapping/mapping_rmse_summary.png)', '']
    report += [f"- [{r['trial']} 관절 배포 입력 매핑](figures/mapping/{r['trial']}_joint_mapping.png) · [발 힘 시간 이동 전후](figures/mapping/{r['trial']}_foot_force_before_after.png)" for r in results]
    if excluded:
        report += ['', '## 비교 제외', ''] + [f"- `{item['id']}`: {item['reason']}" for item in excluded]
    (directory/'analysis.md').write_text('\n'.join(report)+'\n')
    with (directory/'metrics'/'metrics.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['trial','mode','channel','quantity','unit','n','rmse','mae','bias','correlation'])
        writer.writeheader(); writer.writerows(rows)
