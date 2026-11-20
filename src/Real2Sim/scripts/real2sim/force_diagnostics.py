#!/usr/bin/env python3
"""Explain full-replay foot-force mismatch using recorded physical diagnostics."""
import json
from pathlib import Path
import sys

import numpy as np

from real2sim_converter import ROOT, restore_output_owner
sys.path.insert(0, str(ROOT / 'src/Real2Sim/replay'))
from build_data import rolling_rmse


def write_diagnostics(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = output / 'diagnostics'; directory.mkdir(exist_ok=True)
    lines = ['# 전체 LowCmd 재생의 발 힘 차이 진단', '',
        '현재 converter 전체 재생은 실제 정책이 출력했던 LowCmd를 시뮬레이터에 순서대로 넣는다. '
        '시뮬레이터의 상태를 실제 정책에 다시 입력하는 폐루프 제어가 아니며 시작에서만 한 번 초기화한다.', '',
        '## 확인된 현상', '',
        '| 시행 | 45° 이상 기울기 지속 시작 (s) | 시뮬 최대 기울기 (°) | 실제 최대 기울기 (°) | 비발 지지력 평균 (N) |',
        '|---|---:|---:|---:|---:|']
    records = []
    for path in sorted((output / 'replay_full').glob('*_metadata.json')):
        meta = json.loads(path.read_text()); diag = meta['diagnostics']; trial = path.stem.replace('_metadata', '')
        records.append((trial, diag))
        first = diag['sustained_tilt_over_45deg_first_s']
        first_text = f'{first:.3f}' if first is not None else '없음'
        lines.append(f'| {trial} | {first_text} | {diag["max_replay_tilt_deg"]:.2f} | '
                     f'{diag["max_real_tilt_deg"]:.2f} | {diag["steady_mean_nonfoot_support_N"]:.2f} |')
        with np.load(path.with_name(f'{trial}_predictions.npz')) as z:
            t=z['t'][0]; real_q=z['real_q'][0]; sim_q=z['replay_q'][0]
            real_quat=z['real_base_quaternion'][0]; sim_quat=z['replay_base_quaternion'][0]
            force=z['force_world_N'][0, :, :, 2]; nonfoot=z['replay_nonfoot_support_N'][0]
            heights=z['replay_foot_height_m'][0]; valid=z['valid'][0]
            _, error = rolling_rmse(real_q, sim_q, valid)
        tilt = lambda q: np.rad2deg(np.arccos(np.clip(1 - 2*np.sum(q[:,1:3]**2,axis=1),-1,1)))
        fig, axes = plt.subplots(4,1,figsize=(14,12),sharex=True)
        axes[0].plot(t,tilt(real_quat),label='Real body tilt')
        axes[0].plot(t,tilt(sim_quat),label='Replay body tilt')
        axes[0].axhline(45,color='#b76c55',ls=':',label='45 deg'); axes[0].set_ylabel('Tilt [deg]')
        axes[1].plot(t,force.sum(axis=1),label='Feet support')
        axes[1].plot(t,nonfoot,label='Non-foot support')
        axes[1].axhline(diag['nominal_weight_N'],color='gray',ls=':',label='Nominal weight')
        axes[1].set_ylabel('Simulated support [N]')
        for foot,leg in enumerate(('FR','FL','RR','RL')): axes[2].plot(t,heights[:,foot],label=leg)
        axes[2].set_ylabel('Foot bottom height [m]')
        axes[3].plot(t,error,label='12-joint q RMSE, trailing 0.5s');axes[3].set_ylabel('RMSE [rad]')
        for ax in axes:
            if first is not None:ax.axvline(first,color='#b76c55',ls='--',lw=.8)
            ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper left')
        axes[-1].set_xlabel('Real record time [s]')
        fig.suptitle(f'{trial}: full command replay loses balance; foot force omits non-foot ground support\nNo state resets, no physical parameter correction. Different sensor units do not explain missing contact.')
        fig.tight_layout();fig.savefig(directory/f'{trial}_force_diagnosis.png',dpi=150,bbox_inches='tight');plt.close(fig)
    lines += ['', '45°가 100ms 이상 지속되는 구간의 시작을 기록했다. 실제 기울기는 IMU quaternion, '
              '시뮬레이션 기울기는 free-base quaternion으로 계산했다. 자세 임계값은 진단용이며 '
              '하드웨어 낙상 검출기의 검증 기준은 아니다.', '',
              '## 왜 주황색 발 힘이 0에 가까워지는가', '',
              '재생 로봇이 옆으로 기울면 한쪽 발은 공중에 뜨고, hip/thigh 등 발 이외의 충돌 형상이 바닥에 닿는다. '
              'UI의 발 힘은 발 충돌 형상만 합산하므로 다리 옆면의 지지력을 포함하지 않는다. '
              '따라서 실제는 계속 걷는데 재생 모델은 옆면으로 지지하는 상황을 비교하게 된다.', '',
              '1회차에서는 FL/RL의 평균 발 바닥 높이가 약 0.13m이고, FR/RR 쪽 다리 형상이 바닥에 닿는다. '
              '전체 발 지지력과 비발 지지력을 합하면 명목 무게 약 149N에 가까워진다. '
              '즉 힘의 부호나 채널을 단순히 뒤집으면 해결되는 현상이 아니다.', '',
              '## 원인과 미확정 부분', '',
              '확인한 직접 원인은 **독립 재생 모델의 균형 상실과 지지 접촉 부위 변화**다. '
              '실제 로봇의 자세를 유지시킨 명령은 실제 센서 피드백에서 생성된 것이므로, '
              '그 명령이 다른 상태에 있는 시뮬레이터의 균형도 유지한다는 보장은 없다. '
              '초기 몸체 높이/선속도 추정과 명목 관성·모터·접촉 모델의 차이가 영향을 줄 수 있으나, '
              '어느 파라미터가 얼마만큼 기여했는지는 현재 진단만으로 확정하지 않는다.', '',
              '입력 관절 순서와 action scale의 배포 왕복 검증이 통과했다는 사실은 물리적 균형 유지의 증명이 아니다. '
              '데이터를 숨기거나 매 순간 실제 자세를 덮어쓰지 않고, 전체 자유 재생을 그대로 유지했다.', '',
              '## RMSE 표시', '',
              '- UI의 새 그래프: q/dq/ddq별 과거 0.5초 이동창 RMSE. 선택 관절과 12관절 통합을 함께 표시한다.',
              '- 초기 복사 상태/누락 구간은 계산에서 제외한다. 원본의 시간별 그래프는 전체 유효 구간, '
              '오른쪽 요약은 정상 보행 구간이라 집계 범위가 다르다.',
              '- 실제 발 센서는 counts, 시뮬레이션 힘은 N이므로 N 단위 절대 힘 RMSE는 계산하지 않는다.',
              '- 발별 z-RMSE는 각 신호를 따로 평균 0/표준편차 1로 만든 뒤 계산한 파형 오차다. '
              '힘의 크기·영점 오차를 제거하므로 센서 보정 또는 절대 GRF 정확도를 나타내지 않는다.', '']
    for trial,_ in records:lines += [f'[{trial} 자세·지지력·RMSE 진단](diagnostics/{trial}_force_diagnosis.png)', '']
    (output/'foot_force_diagnosis.md').write_text('\n'.join(lines)+'\n')
    restore_output_owner(output)


if __name__ == '__main__':
    write_diagnostics(ROOT / 'src/Real2Sim/result/figure_after')
