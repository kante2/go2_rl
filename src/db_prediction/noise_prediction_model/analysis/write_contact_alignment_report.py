#!/usr/bin/env python3
"""Write the Korean audit report and freshness/native-update sensitivity."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import numpy as np
import pandas as pd
import noise_prediction_model.analysis.audit_contact_alignment_impact as audit
import noise_prediction_model.analysis.landing_event_dynamics as ld
from noise_prediction_model.paths import resolve_recorded_path

ROOT,OUT=audit.ROOT,audit.OUT


def native_changes():
    rows=[]
    for raw in sorted((ROOT/'data/1007').glob('*/*.npz')):
        with np.load(raw,allow_pickle=False) as z:
            t,ff,dq=z['ls_t'],z['ls_ff'],z['ls_dq']
            move=dict(zip(z['ev_label'].astype(str),z['ev_t']))
        for j,leg in enumerate(('FR','FL','RR','RL')):
            ids=np.flatnonzero(np.r_[False,np.diff(ff[:,j])!=0])
            ids=ids[(t[ids]>=move['move_start'])&(t[ids]<=move['move_stop'])]
            intervals=np.diff(t[ids])*1000
            # Exclude long unchanged values in air/stance from short-update statistic.
            short=intervals[(intervals>0)&(intervals<=20)]
            qids=np.flatnonzero(np.r_[False,np.any(np.diff(dq[:,3*j:3*j+3],axis=0)!=0,axis=1)])
            qids=qids[(t[qids]>=move['move_start'])&(t[qids]<=move['move_stop'])]
            qdiff=np.diff(t[qids])*1000;qdiff=qdiff[(qdiff>0)&(qdiff<=20)]
            rows.append(dict(split=raw.parent.name,trial=raw.stem,leg=leg,n_force_value_changes=len(ids),
                n_short_force_intervals=len(short),force_value_change_interval_median_ms=np.median(short) if len(short) else np.nan,
                dq_value_change_interval_median_ms=np.median(qdiff) if len(qdiff) else np.nan))
    return pd.DataFrame(rows)


def main():
    e=pd.read_csv(OUT/'phase_event_features.csv')
    phase=pd.read_csv(OUT/'phase_timing_by_trial_leg.csv')
    corr=pd.read_csv(OUT/'fractional_and_event_shift_correlations.csv')
    search=pd.read_csv(OUT/'rate_search_window_sensitivity.csv')
    fixed=pd.read_csv(OUT/'fixed_shift_correlations_by_trial.csv')
    amp=pd.read_csv(OUT/'phase_amplitude_noise_correlations.csv')
    clock=pd.read_csv(OUT/'source_clock_checks.csv')
    native=native_changes();native.to_csv(OUT/'native_value_change_intervals.csv',index=False)
    sensitivity=[]
    for split,g in e.groupby('split'):
        masks=dict(all=np.ones(len(g),bool),strict_freshness=(g.max_ls_age_ms<=4)&(g.max_sp_age_ms<=6),
                   sync_candidate=g.synchronization_status.eq('CONSISTENT_CANDIDATE'),
                   no_other_detected_onset=g.other_landings_in_extended_window.eq(0))
        for name,mask in masks.items():
            gg=g[mask]
            sensitivity.append(dict(split=split,subset=name,n_events=len(gg),
                fraction_abs_rate_delta_le4ms=gg.rate_delta_ms.abs().le(4).mean(),
                fraction_late_peak_after20ms=gg.late_peak_after20ms.mean()))
    pd.DataFrame(sensitivity).to_csv(OUT/'freshness_and_overlap_sensitivity.csv',index=False)
    target=phase[phase.trial.eq('1.5_3')].set_index('leg').loc[list(ld.LEGS)]
    target_rl=e[e.trial.eq('1.5_3')&e.leg.eq('RL')]
    target_rr=e[e.trial.eq('1.5_3')&e.leg.eq('RR')]
    def number(x):return f'{x:.3f}'
    lines=['# 접촉력·발 수직 속도 시각 불일치와 소음 상관관계 검토','',
        '## 결론','',
        '`1.5_3`에서 사용자가 본 차이는 반복된다. 특히 뒷발은 초기 급감속 뒤 약 40~44ms에 더 큰 힘 피크가 나타나는 두 단계 파형이다. '
        '초기 힘 증가·속도 변화의 시각 차이는 대부분 수 ms이지만, 최대 하중 시점은 그보다 훨씬 늦다. '
        '이는 전처리의 단일 고정 시계 오프셋으로 설명되지 않는다. 토크/모델 기반 힘에서도 늦은 하중 피크가 보이므로 '
        'foot-force 센서만의 지연이라고 단정할 근거도 없다.','',
        '다른 train 시퀀스로 정한 작은 시간차를 적용해도 전체 소음 상관이 크게 회복되지 않는다. '
        '반면 초기 충돌과 후반 하중 중 어느 단계를 특징으로 정의하는지, 소음 RMS 포락선의 후속 반응까지 비교하는지에 따라 상관계수는 달라진다. '
        '기존 결과를 곧바로 “힘·속도가 소음에 무관하다”로 해석하면 안 된다.','',
        '## 조사 범위와 불변 조건','',
        f'- raw train/test 51개 파일 전체를 확인했다. 기존 이벤트 기준으로 분석 가능한 48개 시퀀스, {len(e):,}개 이벤트 '
        f'(train {e.split.eq("train").sum():,}, test {e.split.eq("test").sum():,})를 사용했다.',
        '- train의 6,730개 이벤트 ID와 기존 힘 피크·직전 속도·PCM 후 40ms RMS 값을 재대조했다.',
        '- 기존 ON=10/OFF=5, 6ms 지속 검출과 원래 최초 통과 시각을 그대로 사용했다. 0ms는 물리적 최초 접촉의 정답이 아니다.',
        '- `left_0.1_1`, `right_0.3_3`, `test`는 힘의 바닥값 때문에 반복 OFF 검출이 없어 이벤트가 0개다. '
        '이들에서도 CSV/raw 매핑은 검증했지만 착지 반복 분석에는 포함할 수 없었다. `trial_coverage.csv`에 기록했다.',
        '- 이벤트 창은 원래 [-50,+100]ms, 2ms 표본이다. 데이터·기존 전처리·기존 그림·모델을 바꾸지 않았다.',
        '- 모든 시간 이동은 이 폴더의 민감도 분석에만 적용했다. 오디오에 맞추어 시계나 이벤트를 보정하지 않았다.','',
        '## 1. 1.5_3에서 실제로 반복되는 현상','',
        '피크는 고정 검색 구간의 최대값 동률 시각 중간값이다. 경계/동률 폭도 CSV에 남겼다. '
        '아래의 최대 힘 시각은 [0,+60]ms에서 구했다. 초기 속도 변화율은 [-12,+20]ms의 12ms centered 차분 최대다.','',
        '|발|이벤트 수|초기 속도 변화율 피크|최대 힘 시각|최대 힘이 +20ms 뒤인 비율|후반/초기 최대 힘 비율|',
        '|---|---:|---:|---:|---:|---:|']
    for leg,r in target.iterrows():
        lines.append(f'|{leg}|{r.n_events}|{r.initial_deceleration_peak_median_ms:+.1f}ms|'
                     f'{r.sensor_peak_0_60_median_ms:+.1f}ms|{r.fraction_peak_after20ms:.1%}|{r.late_to_early_ratio_median:.2f}|')
    lines+=['','비율은 각 이벤트의 초기 [0,12]ms 최대값 대비 후반 [20,60]ms 최대값을 구한 뒤 중앙값이다. '
        'RL은 43개 중 39개, RR은 42개 중 38개에서 최대 힘이 +20ms 이후였다. '
        '시퀀스 중앙의 연속 4개 이벤트를 소음과 무관하게 선택한 그림에도 같은 패턴이 보인다.','',
        '- [RL 개별 이벤트](1.5_3_RL_individual_events.png), [RR 개별 이벤트](1.5_3_RR_individual_events.png)',
        '- [FL 개별 이벤트](1.5_3_FL_individual_events.png), [FR 개별 이벤트](1.5_3_FR_individual_events.png)',
        '- [기록 전체 반복 시각](trials/train/1.5_3_timing_and_repetition.png): 맨 아래는 각 착지의 실제 시간 순서 산점도다.','',
        '그림의 1행은 5ms/20ms 소음 RMS, 2행은 센서 힘과 추정 발 수직 속도, 3행은 같은 20ms 필터를 적용한 센서 힘과 토크 기반 추정 힘, '
        '4행은 정규화한 두 변화율이다. 소음 행의 다른 발 이름은 그 발의 검출 시각이다. '
        '다른 단위의 축을 사용하므로 3행 두 선의 높이가 같다는 것으로 힘 크기가 일치한다고 판단하면 안 된다.','',
        '기존 [0,+40]ms 최대 힘 창도 일부 후반 피크를 자른다. [0,+60]ms로 확인한 피크가 +40ms 뒤인 비율은 '
        f'RL {target_rl.force_peak_0_60_ms.gt(40).mean():.1%}, '
        f'RR {target_rr.force_peak_0_60_ms.gt(40).mean():.1%}다. '
        '따라서 기존 `force_post_peak`는 항상 그 착지의 전체 최대 하중을 뜻하지 않는다. [0,80]ms 피크 시각도 따로 남겼으며 이 시퀀스에서는 유사했다.','',
        '## 2. 다른 시퀀스와 검색 구간 민감도','',
        f'이벤트가 20개 이상인 trial×발 조합 {len(phase[phase.n_events.ge(20)])}개 중 '
        f'{len(phase[phase.n_events.ge(20)&phase.fraction_peak_after20ms.ge(.8)])}개에서 80% 이상의 이벤트가 늦은 힘 피크를 보였다. '
        '대부분 1.2/1.5/1.8 이름의 전진 보행 뒷발이며, test 1.2_7/1.5_7/1.8_7에서도 재현된다. '
        '파일명의 숫자를 측정 속도로 해석하지 않는다. 이벤트별 측정 평면 몸체 속도는 별도 열에 저장했다.','',
        '이전 검토의 “힘·속도 변화율 피크가 ±4ms 이내 약 92%”는 **초기 [-12,+20]ms**를 검색한 결과다. '
        '그 뒤의 하중/반동을 포함한 전체 과정의 정렬을 보증하지 않는다. 검색을 넓히면 다음처럼 바뀐다.','',
        '|split|검색 구간|속도 변화율−힘 변화율 피크 차이 중앙값|±4ms 이내|', '|---|---|---:|---:|']
    for (split,hi),g in search.groupby(['split','search_hi_ms']):
        lines.append(f'|{split}|[{-12 if hi==20 else -20},+{hi}]ms|{g.rate_delta_ms.median():+.1f}ms|{g.rate_delta_ms.abs().le(4).mean():.1%}|')
    lines+=['','넓은 창에서 다른 물리적 단계의 피크가 선택될 수 있으므로 차이가 커졌다는 것 역시 시계 오차의 증거는 아니다. '
        '파형은 두 단계인데 하나의 피크 차이로 요약하는 방법 자체에 한계가 있다.','',
        '## 3. 전처리·센서 표본·속도 계산 점검','',
        '51개 CSV의 모든 행에서 force, q, dq, gyro, quaternion이 같은 원본 LowState previous 표본과 일치함을 확인했다. '
        'Sport 몸체 속도도 원본 previous 표본과 일치했다. force와 관절 데이터 사이에 따로 적용한 시간 이동은 없다. '
        '수신 시각의 일치는 센서 내부 획득/필터 지연의 동일성을 보증하지 않는다.','',
        f'원본 값이 실제로 달라지는 짧은 간격(20ms 이하)의 trial×발 중앙값은 foot force '
        f'{native.force_value_change_interval_median_ms.median():.2f}ms, dq '
        f'{native.dq_value_change_interval_median_ms.median():.2f}ms다. '
        '힘이 계단형으로 갱신되는 반면 속도는 더 자주 바뀌어서 몇 ms의 피크 차이가 생기기 쉽다. '
        '같은 값을 재측정했을 가능성이 있으므로 이를 센서의 실제 취득 주파수라고 단정하지 않는다.','',
        '속도는 `R(v_body + ω×p + J dq)`로 분해했다. train에서 접촉 전후 −12→+12ms 성분 변화의 절댓값 중앙값은 '
        f'몸체 {e[e.split.eq("train")].body_vz_prepost_change_mps.abs().median():.3f}, '
        f'회전 {e[e.split.eq("train")].rotation_vz_prepost_change_mps.abs().median():.3f}, '
        f'관절 {e[e.split.eq("train")].joint_vz_prepost_change_mps.abs().median():.3f} m/s다. '
        '급변의 주성분은 관절 운동학이다. 이 수치는 분산 기여율이나 인과 기여도는 아니다.','',
        f'Sport 몸체 속도를 previous 대신 선형 보간한 진단에서 원래 발 속도와의 이벤트별 MAE 중앙값은 train '
        f'{e[e.split.eq("train")].vz_sport_linear_mae_mps.median():.4f}m/s였다. '
        '초기 피크 정렬 비율도 거의 변하지 않았다. 이는 Sport previous 보간만으로 40ms의 후반 하중을 설명하기 어렵다는 뜻이다. '
        '이 보간은 미래 표본을 쓰는 진단이며 원래 계산에는 반영하지 않았다.','',
        '원본 불규칙 시각의 q를 중앙 7표본 2차식으로 미분해 dq 대신 사용한 대안도 비교했다. '
        f'발 속도 MAE 중앙값은 train {e[e.split.eq("train")].vz_qdot_fit_mae_mps.median():.3f}m/s이며 '
        '초기 변화율 피크 차이 중앙값은 −2ms였다. 내부 dq 필터와 대안 미분 모두 오차를 가질 수 있고, 독립 속도 정답이 없어서 어느 쪽이 진실인지 정할 수 없다.','',
        'CSV 전체에서 허용한 source age는 최대 20ms였다. 더 엄격한 이벤트 조건(LowState 최대 age≤4ms, Sport≤6ms)으로 '
        '제한해도 `1.5_3` RL 40개 중 36개, RR 37개 중 33개가 +20ms 이후 힘 피크였다. '
        '신선한 표본만으로도 반복되는 현상이다. 4/8/12/20ms 차분 민감도도 별도 CSV에 저장했다.','',
        '토크 기반 Fz의 `1.5_3` 피크 중앙값은 RL +38ms, RR +36ms였다. '
        '모터/URDF 추정에도 후반 하중이 나타나는 점은 센서만 통째로 지연되었다는 설명을 약화시킨다. '
        '그러나 이 추정도 마찰·구동계·IMU·모델 오차가 있고 관절 상태를 공유한다. 실제 힘의 정답이나 센서 보정 근거가 아니다.','',
        '## 4. 소음 상관관계에 미친 영향','',
        '### A. 작은 공통 시간차만 수정하면 회복되는가?','',
        '평가하는 train trial을 제외한 나머지 train trial에서 발별 초기 변화율 피크 차이의 trial 중앙값들을 구하고, '
        '그 중앙값으로 이동량을 정했다. test는 train 전체로만 이동량을 정했다. 소음은 사용하지 않았다. '
        '2ms 격자 반올림 결과뿐 아니라 실제 중앙값으로 선형 보간한 fractional 결과도 확인했다. '
        '아래는 후자이며 검색 경계 피크 이벤트를 제외했다.','',
        '각 이벤트에서 [-20,+60]ms의 동일한 41개 시각을 대상으로 평균 동률 순위를 매긴 Spearman을 계산했다. '
        '각 trial 이벤트 중앙값 → trial 동일 가중 평균 순서다. 단일 pooled correlation이 아니다. '
        '작은 이벤트 수의 trial도 포함하므로 `n_events`를 함께 확인해야 한다.','',
        '|split|변수|원래 5ms RMS와 ρ|다른 train의 힘·속도 시간차만 적용한 ρ|', '|---|---|---:|---:|']
    cc=corr[corr.strategy.eq('other_train_fractional')&corr.audio.eq('noise_5ms_dbfs')]
    for (split,driver),g in cc.groupby(['split','driver']):
        lines.append(f'|{split}|{driver}|{g.rho_original.mean():.3f}|{g.rho_shifted.mean():.3f}|')
    lines+=['','전체적으로 큰 회복은 없었다. 개별 이벤트의 변화는 존재하며, 같은 이벤트의 힘·속도 시간차로 각각 이동하는 '
        '회고적 민감도 분석도 수행했다. 그 경우에도 전체 상관은 일관되게 좋아지지 않았다. '
        '해당 이벤트로 정한 이동량은 배포 가능한 시계 보정값이 아니다.','',
        '### B. 초기 충격 뒤 따라오는 소음 포락선은 놓치고 있었는가?','',
        '고정 이동량 −20/−10/−6/−4/−2/0/+2/+4/+6/+10/+20ms를 모두 조사했다. '
        '`s=-20ms`는 기계 신호 `x(t−20ms)`를 소음 `noise(t)`와 비교하므로 **기계 변화 뒤 20ms의 소음**을 보는 것이다. '
        '아래는 같은 원래 이벤트 전체로 계산한 train의 trial 중앙값 동일 가중 평균이다.','',
        '|변수|동시각 ρ|기계 신호 뒤 20ms 소음과 ρ|', '|---|---:|---:|']
    for driver in audit.DRIVERS:
        g=fixed[fixed.split.eq('train')&fixed.driver.eq(driver)&fixed.audio.eq('noise_5ms_dbfs')]
        lines.append(f'|{driver}|{g[g.sample_shift_ms.eq(0)].median_rho.mean():.3f}|'
                     f'{g[g.sample_shift_ms.eq(-20)].median_rho.mean():.3f}|')
    lines+=['','변화율과 소음은 동시각 비교보다 후속 포락선을 비교할 때 더 큰 ρ를 보일 수 있다. '
        '그러나 힘·속도 원값의 상관은 같은 이동으로 떨어졌다. 모든 신호에 통하는 하나의 시계 보정이라는 해석과 맞지 않는다. '
        'RMS 포락선의 형상, 진동·잔향, 다른 발의 접촉, 모터 소음도 이런 차이를 만들 수 있다. '
        '20ms가 진짜 센서 지연이라는 뜻이 아니며 가장 좋은 상관을 기준으로 데이터를 재정렬하지 않았다. '
        '다른 고정 이동량과 20ms RMS 비교는 [민감도 그림](noise_correlation_shift_sensitivity.png)에 있다.','',
        '### C. 최대 힘과 직전 속도라는 특징 정의도 영향을 주는가?','',
        '각 trial×발에서 이벤트 간 특징 크기와 원본 PCM [0,40)ms RMS의 Spearman을 별도로 구했다. '
        '유효 이벤트 10개 이상이며 상수가 아닌 조합만 사용한 뒤 조합별 ρ의 중앙값을 표시한다. '
        '이 계산은 A/B의 이벤트 내부 파형 상관과 질문 자체가 다르다.','',
        '|특징|train ρ 중앙값|test ρ 중앙값|','|---|---:|---:|']
    for feature in ('force_peak_0_40','force_early_0_12_max','force_late_20_60_max',
                    'downward_minus2_mps','downward_pre20_max_mps','downward_pre20_mean_mps'):
        g=amp[amp.feature.eq(feature)&amp.target.eq('noise_post40_rms_dbfs')]
        lines.append(f'|{feature}|{g[g.split.eq("train")].rho.median():.3f}|{g[g.split.eq("test")].rho.median():.3f}|')
    lines+=['','`1.5_3`의 RR은 힘 [0,40]ms 피크 ρ=0.227에서 후반 [20,60]ms 피크 ρ=0.586으로 달라졌으나, '
        'RL은 0.296→0.110이었다. 일부 발의 상관을 높이는 정의가 전체 데이터의 더 옳은 물리량이라는 뜻은 아니다. '
        '특징의 시간 범위가 비교 결과를 바꾼다는 확인이며 탐색적 결과다. 후반 힘은 소음 발생 뒤의 값도 포함하므로 접촉 전 예측 입력으로 쓰면 미래 정보다.','',
        f'train에서 마지막 −2ms 하강 속도 / 이전 20ms 최대 하강 속도 비율의 이벤트 중앙값은 '
        f'{e[e.split.eq("train")].downward_minus2_over_pre20max.median():.3f}였다. '
        '`1.5_3`은 FL 0.741, FR 0.640, RL 0.582, RR 0.698이다. '
        '기존 직전 표본은 이전 창 최대치보다 이미 작아진 경우가 많다. 힘 임계값 검출보다 물리적 접촉이 앞섰을 가능성과 속도 파형 변동을 포함한다. '
        '이전 20ms 최대값도 물리적 충돌 속도의 정답은 아니다.','',
        '## 물리적 해석과 다음 분석 방향','',
        '발의 속도가 가장 클 때 힘도 가장 커야 하는 법칙은 없다. 충돌에서 힘의 적분은 운동량 변화와 연결되지만, '
        '접촉 후 발이 거의 멈춰도 몸체 하중과 제어 토크를 전달하며 힘이 커질 수 있다. '
        '특히 이 발 속도는 강체 URDF foot-frame 기준점의 추정치이며 변형하는 실제 발바닥 접촉점 속도가 아니다. '
        '전체 다물체·접촉 제약을 무시하고 같은 발에 `F=m_foot*dv_foot/dt`를 직접 적용하면 안 된다.','',
        '지금은 **단일 시간 이동으로 정렬**하기보다 **초기 급감속 구간과 후반 하중 구간을 분리**하고, '
        '초기 변화량/상승 시간/적분 및 후반 최대 힘을 서로 다른 특징으로 남기는 방향이 타당하다. '
        '전/후 소음 에너지와 시간 응답을 비교하되, 접촉 전에 예측할 특징과 접촉 후 설명할 특징을 구분해야 한다.','',
        f'단일 마이크이고 확장 창에 다른 검출 착지가 없는 비율은 train '
        f'{e[e.split.eq("train")].other_landings_in_extended_window.eq(0).mean():.1%}, test '
        f'{e[e.split.eq("test")].other_landings_in_extended_window.eq(0).mean():.1%}뿐이다. '
        '다른 발·모터·바닥 진동·잔향이 섞이므로 발별 인과 기여도나 중요도를 이 상관계수만으로 확정할 수 없다. '
        '낮은 힘 바닥값 때문에 놓친 접촉도 있으므로 “다른 검출 없음”이 실제 단독 충돌을 보증하지 않는다. '
        '중첩·자기상관을 무시한 p-value나 신뢰구간은 보고하지 않았다.','',
        '## 산출물과 재현','',
        '- [전체 개요](alignment_overview.png), [두 단계 힘/소음](two_stage_force_and_noise.png)',
        '- `event_timing_and_features.csv`, `phase_event_features.csv`: 모든 이벤트 및 단계별 특징.',
        '- `timing_by_trial_leg.csv`, `phase_timing_by_trial_leg.csv`: 시퀀스/발별 반복 비율·시각.',
        '- `rate_search_window_sensitivity.csv`, `timing_estimator_sensitivity.csv`: 검색창/차분/속도 추정 대안.',
        '- `noise_correlations_paired_by_event.csv`, `fractional_and_event_shift_correlations.csv`: 시간 이동 전후 상관.',
        '- `phase_amplitude_noise_correlations.csv`: 특징/소음 구간별 이벤트 간 상관.',
        '- `source_clock_checks.csv`, `native_value_change_intervals.csv`, `freshness_and_overlap_sensitivity.csv`: 원본/표본 신선도 검토.',
        '- `audited_waveforms.npz`, `force_crosscheck_waveforms.npz`: 그림과 수치의 정확한 입력 배열.',
        '- `trials/train`, `trials/test`: 이벤트가 있는 48개 시퀀스 각각의 반복 그림.',
        '- `verification.json`, `phase_verification.json`, `report_verification.json`: 입력 해시와 검증.',
        '', '```bash',
        '.venv/bin/python noise_prediction_model/analysis/audit_contact_alignment_impact.py',
        '.venv/bin/python noise_prediction_model/analysis/summarize_contact_alignment_impact.py',
        '.venv/bin/python noise_prediction_model/analysis/write_contact_alignment_report.py',
        '.venv/bin/python -m unittest discover -s noise_prediction_model/analysis -p test_contact_alignment_impact.py -v',
        '```','']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    original=json.loads((OUT/'verification.json').read_text())
    for path,value in original['input_sha256'].items():
        if ld.digest(resolve_recorded_path(ROOT/path))!=value:raise AssertionError(f'Protected input changed: {path}')
    result=dict(passed=True,protected_files_rechecked=len(original['input_sha256']),
                source_csv_rows_verified=int(clock.n_csv_rows.sum()),n_native_trial_leg_checks=len(native),
                script_sha256=ld.digest(__file__),report_sha256=ld.digest(OUT/'report.md'))
    (OUT/'report_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
