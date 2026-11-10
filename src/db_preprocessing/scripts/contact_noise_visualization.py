"""Force-only contact estimates over measured noise; never align using audio.

These are threshold-based loaded-contact estimates, not ground-truth gait labels.
Estimated intervals are backdated to the first threshold crossing only after
confirmation. The causal state and confirmation time are saved separately.
"""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/low_noise_matplotlib')
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
LEGS=('fl','fr','rl','rr')
DEFAULT_ON=10.
DEFAULT_OFF=5.
DEFAULT_CONFIRM_MS=6.


from contact_detection import detect_contact


def intervals(times,states,segments):
    if not len(times):return []
    dt=float(np.median(np.diff(times))) if len(times)>1 else .002
    changes=np.r_[True,(np.diff(states)!=0)|(np.diff(segments)!=0)|(np.diff(times)>1.5*dt+1e-9)]
    starts=np.flatnonzero(changes);ends=np.r_[starts[1:],len(times)]
    return [(float(times[a]),float(times[b-1]+dt),int(states[a])) for a,b in zip(starts,ends)]


def plot_contact_noise(frame,trial_name,output_dir,on=DEFAULT_ON,off=DEFAULT_OFF,
                       confirm_ms=DEFAULT_CONFIRM_MS,zoom_seconds=1.5,make_plot=True,
                       raw_path=None,split=None):
    """Per-foot force/kinematic speed plots and contact-event CSV artifacts."""
    from contact_event_extraction import analyze_frame,save_trial_result
    if raw_path is None:
        matches=list((ROOT/'data/1104').glob(f'*/{trial_name}.npz'))
        if len(matches)==1:raw_path=matches[0]
    if split is None:split=Path(raw_path).parent.name if raw_path is not None else 'unknown'
    output_dir=Path(output_dir)
    sync_path=output_dir/'synchronization.json'
    sync=json.loads(sync_path.read_text()) if sync_path.exists() else {}
    result=analyze_frame(frame,trial_name,on,off,confirm_ms,raw_path=raw_path,split=split,sync_summary=sync)
    save_trial_result(result,trial_name,output_dir,on,off,confirm_ms,make_plot,zoom_seconds)
    return result['summary']


def save_contact_summary(figure_dir,rows):
    figure_dir=Path(figure_dir);figure_dir.mkdir(parents=True,exist_ok=True)
    summary=figure_dir/'contact_summary.csv'
    names={row['trial'] for row in rows}
    previous=pd.read_csv(summary) if summary.exists() else pd.DataFrame()
    if len(previous):previous=previous[~previous.trial.isin(names)]
    pd.concat([previous,pd.DataFrame(rows)],ignore_index=True).to_csv(summary,index=False)
    (figure_dir/'contact_readme.md').write_text('''# 발 접촉 이벤트·접촉력·접촉 직전 속도·소음

contact_noise.png는 기존처럼 소음과 FL/FR/RL/RR 접촉 상태를 보여준다. 전체 보행과 중앙 확대를 위아래로 배치한다.
contact_force_velocity_with_noise.png는 맨 위에 공유 마이크의 20ms RMS 소음을 추가하고, 아래 발별 힘·속도와 같은 시간축으로 표시한다.
두 그림의 확대 구간은 동일하며 소음과 무관하게 선택한다.
발별 색은 접촉력, 빨간 선은 지면 고정 좌표계의 수직 속도 v_z이며 양수는 상승, 음수는 하강이다. 빨간 ▼는 접촉 직전 마지막 표본이다.
속도 선은 착지 전후와 발을 들어 올리는 구간 전체를 표시하며, CSV의 하강 속도 max(-v_z,0) 열과 구분한다.
색칠한 영역은 기본 접촉 전 20ms~후 40ms이며 같은 영역의 시계열이 contact_event_samples.csv에 저장된다.

Go2 URDF의 각 발 위치 p(q)와 Jacobian J(q)를 사용한다.
발 속도 = R_WB * (Sport 몸체 선속도 + LowState gyro × p(q) + J(q) * LowState dq).
원본 위치 변화와 비교한 좌표계 진단에서 본 녹음의 Sport 선속도는 body-frame 해석과 잘 맞았다.
공식 SDK 예제의 odometry-frame 주석과 다르므로 contact_detection.json의 body_velocity_frame_audit에 두 가설의 오차를 남긴다.
sp_foot_vel은 속도 계산에 사용하지 않는다. 공식 URDF 출처/라이선스는 preprocessing/models에 있다.
속도는 URDF foot-frame 기준 강체 운동학 추정치이며 변형하는 발바닥의 실제 접촉점 속도가 아니다.

1104에서는 동기화 진단에 기록한 발별 시작 구간 임계값을 재사용한다. 설정이 없으면 ON=10, OFF=5를 사용한다. 단위는 보정된 N이 아닌 센서 값이며 6ms 지속 후 접촉을 확정한다.
최초 통과 시각과 확정 시각을 구분하고, 공백·segment 경계에서는 상태를 초기화한다.
contact_states.csv의 estimated_contact는 회고적 상태, causal_contact는 그때까지 확정된 상태다.
초기 접촉 상태만으로 착지 이벤트를 생성하지 않는다. 낮은 힘이 공중 상태를 증명하지 않는다.

각 split의 contact_event_features.csv는 발 접촉 하나당 한 행이다.
접촉 시각/힘, 후 40ms 접촉력 피크·증가율·적분, 직전 속도 xyz·크기·하강 속도, 원본 PCM 기반 영역 전체/후 40ms RMS·peak를 포함한다.
원본 오디오는 기존 audio_t0/audio_slope와 CSV의 audio_delay_s만 적용한다. 추가 동기화는 하지 않는다.
시계열 samples의 audio_dbfs_20ms는 centered 20ms 값이므로 표시 시각의 전후 10ms를 사용한다.
반면 noise_post_rms_dbfs는 정확히 접촉~+40ms의 원본 파형을 사용한다.

feature_valid=False는 영역이 공백/경계에 걸리거나 직전 속도·오디오를 사용할 수 없다는 뜻이다.
synchronization_review_required는 기존 동기화 진단 상태를 전달한다. quality_ok는 두 조건을 합친 보수적 사용 표시다.
어느 발이든 마이크 소음은 전체 로봇에서 온다. other_leg_contact_count_in_window/other_legs_in_window로 동시 접촉을 표시한다.
force_post_integral_sensor_unit_s는 힘 단위 미보정 적분이며 N·s로 해석하지 않는다.
이벤트 검출·구간 선정에 소음을 쓰지 않지만, 접촉력 피크 등은 접촉 후 특징이므로 접촉 전 예측 입력으로 쓰면 미래 정보가 된다.

실행: .venv/bin/python preprocessing/contact_noise_visualization.py
또는: .venv/bin/python preprocessing/contact_event_extraction.py
부분 갱신: --trials 1.8_7 (기존 다른 trial을 유지하고 같은 trial은 교체)
구간 변경: --before-ms 20 --after-ms 40 (전체를 같은 설정으로 재생성)
임계값 변경: --on-threshold 15 --off-threshold 7 --confirm-ms 6
train/test는 따로 누적하며 test를 train에 섞지 않는다. 기존 시계열 CSV·모델은 변경하지 않는다.
''')


def main():
    from contact_event_extraction import main as extract_events
    extract_events()


if __name__=='__main__':main()
