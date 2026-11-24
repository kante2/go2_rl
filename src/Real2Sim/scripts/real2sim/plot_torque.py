"""Static torque plots from a saved result bundle; no raw-log access."""
import csv
import json
from pathlib import Path
import numpy as np
from .config import JOINTS


def plot_torque(result_dir, directory=None):
    result_dir = Path(result_dir)
    directory = Path(directory) if directory else result_dir / 'figures/torque'
    mode = 'before'  # Torque is identical in physical and actor-coordinate views.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=json.loads((result_dir/'metrics/mapping_summary.json').read_text())
    directory.mkdir(parents=True,exist_ok=True)
    rows=[]; pooled=[]
    for trial in summary:
        name=trial['trial']
        with np.load(result_dir/'processed'/f'{name}_mapping.npz',allow_pickle=False) as z:
            t=z['t'];mask=z['steady'];real=z[f'real_tau_{mode}'];sim=z[f'sim_tau_{mode}']
            metrics=trial['metrics'][mode]['joints']['tau']
            total=trial['metrics'][mode]['tau']
            pooled.append(dict(trial=name,unit='N·m',lag_ms=trial['synchronization']['applied_lag_s']*1000,**total))
            for joint,m in zip(JOINTS,metrics):rows.append(dict(trial=name,joint=joint,unit='N·m',**m))
            start=trial['synchronization']['calibration']['window_s'][0]+1
            for zoom in (False,True):
                view=(t>=start)&(t<=start+2.4) if zoom else np.ones(len(t),bool)
                fig,axes=plt.subplots(4,3,figsize=(16,11),sharex=True)
                for j,ax in enumerate(axes.flat):
                    ax.plot(t[view],real[view,j],color='C0',lw=.8,label='Real estimated motor torque')
                    ax.plot(t[view],sim[view,j],color='C1',lw=.8,label='Sim actuator torque')
                    ax.set_title(f'{JOINTS[j].replace("_joint","")} | RMSE {metrics[j]["rmse"]:.3f} N m',fontsize=10)
                    ax.set_ylabel('N m');ax.grid(alpha=.2)
                for ax in axes[-1]:ax.set_xlabel('Real record time [s]')
                axes[0,0].legend(fontsize=7)
                stage='01: time-aligned baseline' if mode=='before' else '02: deployment joint mapping; torque gain=1, offset=0 (no rule)'
                fig.suptitle(f'{name}: ls_tau | {stage}\nBoth 15 Hz LPF; same time axis/mask; RMSE over recording interior, not zoom window')
                fig.tight_layout();fig.savefig(directory/f'{name}_tau{"_zoom" if zoom else ""}.png',dpi=130);plt.close(fig)
    with (directory/'tau_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['trial','joint','unit','n','rmse','mae','bias','correlation'])
        writer.writeheader();writer.writerows(rows)
    (directory/'tau_summary.json').write_text(json.dumps(dict(mode=mode,source='ls_tau',cutoff_hz=15,
        real_semantics='estimated motor torque',sim_semantics='simulated actuator torque',
        mapping='existing deployment joint order only; validated common time shift; no torque scale/offset fitting',
        pooled=pooled,joints=rows),indent=2,ensure_ascii=False))
    fig,ax=plt.subplots(figsize=(max(9, len(pooled) * .65),6))
    ax.bar([r['trial'] for r in pooled],[r['rmse'] for r in pooled]);ax.set_ylabel('12-joint pooled torque RMSE [N m]')
    ax.set_title(f'{mode}: same recording-interior evaluation samples');ax.grid(axis='y',alpha=.2)
    plt.setp(ax.get_xticklabels(), rotation=60, ha='right')
    fig.tight_layout();fig.savefig(directory/'summary_tau_rmse.png',dpi=150);plt.close(fig)
    text=['# 현실·시뮬 ls_tau 비교','',
        f"대상: real/sim의 {summary[0]['provenance']['dataset']} 폴더 안 {len(summary)}쌍. 같은 관절 순서, 15 Hz 필터와 200 Hz 공통 PC 시간축을 사용한다. 앞 구간 q/dq로 추정하고 독립된 뒤 구간으로 검증한 시행별 공통 이동을 토크에도 적용한다. 검증 미달 시행은 0ms를 유지한다.",
        'real은 모터 추정 토크, sim은 시뮬 액추에이터 토크다. lc_tau(feedforward)는 비교 대상이 아니다.',
        '01은 시간 정렬 기준, 02는 배포 매핑 후다. 배포 정책에 tau 관측이나 토크 보정 계수가 없어 02는 관절 이름 대응만 적용한다. N·m 값과 RMSE는 동일하며 action_scale을 토크에 적용하지 않는다.',
        '진폭/영점 피팅과 재시뮬레이션은 하지 않았다. 시작 2초·끝 1초와 누락 주변을 제외한 녹음 내부 평가 마스크로 통계를 계산하며 확대 창과 별개다.', '',
        '| 시행 | 공통 시간 이동(ms) | 12관절 통합 RMSE(N·m) | MAE(N·m) |', '|---|---:|---:|---:|']
    text += [f"| {r['trial']} | {r['lag_ms']:+.1f} | {r['rmse']:.4f} | {r['mae']:.4f} |" for r in pooled]
    text += ['', '[시행별 RMSE](summary_tau_rmse.png)']
    text += [f"- [{r['trial']} 전체]({r['trial']}_tau.png) · [확대]({r['trial']}_tau_zoom.png)" for r in pooled]
    (directory/'tau_analysis.md').write_text('\n'.join(text)+'\n')
    print(f'{mode}: {len(pooled)} trials / {len(rows)} joint comparisons -> {directory}',flush=True)
