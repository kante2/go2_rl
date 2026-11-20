#!/usr/bin/env python3
"""Compare logged motor torque before/after deployment mapping. No simulation."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'replay'))
from build_data import build, DEFAULT_DATASET, JOINTS, metric


def export(result_dir, before_dir, after_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=json.loads((result_dir/'mapping_summary.json').read_text())
    for mode,directory in (('before',before_dir),('after',after_dir)):
        directory.mkdir(parents=True,exist_ok=True)
        rows=[]; pooled=[]
        for trial in summary:
            name=trial['trial']
            with np.load(result_dir/f'{name}_mapping.npz',allow_pickle=False) as z:
                t=z['t'];mask=z['steady'];real=z[f'real_tau_{mode}'];sim=z[f'sim_tau_{mode}']
                metrics=[metric(real[:,j],sim[:,j],mask) for j in range(12)]
                total=metric(real.reshape(-1),sim.reshape(-1),np.repeat(mask,12))
                pooled.append(dict(trial=name,unit='N·m',**total))
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
                    fig.suptitle(f'{name}: ls_tau | {stage}\nBoth 15 Hz LPF; same time shift/mask; RMSE over steady walking, not zoom window')
                    fig.tight_layout();fig.savefig(directory/f'{name}_tau{"_zoom" if zoom else ""}.png',dpi=130);plt.close(fig)
        with (directory/'tau_metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['trial','joint','unit','n','rmse','mae','bias','correlation'])
            writer.writeheader();writer.writerows(rows)
        (directory/'tau_summary.json').write_text(json.dumps(dict(mode=mode,source='ls_tau',cutoff_hz=15,
            real_semantics='estimated motor torque',sim_semantics='simulated actuator torque',
            mapping='existing deployment joint order only; no torque scale/offset; no fitted parameters',
            pooled=pooled,joints=rows),indent=2,ensure_ascii=False))
        fig,ax=plt.subplots(figsize=(9,4))
        ax.bar([r['trial'] for r in pooled],[r['rmse'] for r in pooled]);ax.set_ylabel('12-joint pooled torque RMSE [N m]')
        ax.set_title(f'{mode}: same steady-walking samples');ax.grid(axis='y',alpha=.2)
        fig.tight_layout();fig.savefig(directory/'summary_tau_rmse.png',dpi=150);plt.close(fig)
        text=['# 현실·시뮬 ls_tau 비교','',
            f"대상: real/sim의 {summary[0]['provenance']['dataset']}/0.2_1.npz~0.2_5.npz. 두 데이터는 같은 관절 순서, 15 Hz 필터, 200 Hz 공통 시간축과 기존 보행 위상 이동을 사용한다.",
            'real은 모터 추정 토크, sim은 시뮬 액추에이터 토크다. lc_tau(feedforward)는 비교 대상이 아니다.',
            '01은 시간 정렬 기준, 02는 배포 매핑 후다. 배포 정책에 tau 관측이나 토크 보정 계수가 없어 02는 관절 이름 대응만 적용한다. N·m 값과 RMSE는 동일하며 action_scale을 토크에 적용하지 않는다.',
            '진폭/영점 피팅과 재시뮬레이션은 하지 않았다. 정상 보행 마스크로 통계를 계산하며 확대 창과 별개다.', '',
            '| 시행 | 12관절 통합 RMSE(N·m) | MAE(N·m) |', '|---|---:|---:|']
        text += [f"| {r['trial']} | {r['rmse']:.4f} | {r['mae']:.4f} |" for r in pooled]
        text += ['', '[시행별 RMSE](summary_tau_rmse.png)']
        text += [f"- [{r['trial']} 전체]({r['trial']}_tau.png) · [확대]({r['trial']}_tau_zoom.png)" for r in pooled]
        (directory/'tau_analysis.md').write_text('\n'.join(text)+'\n')
        print(f'{mode}: {len(pooled)} trials / 60 joints -> {directory}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=PROJECT/'data')
    p.add_argument('--dataset',default=DEFAULT_DATASET)
    p.add_argument('--result-dir',type=Path,default=PROJECT/'result/data_mapping')
    p.add_argument('--before-dir',type=Path,default=PROJECT/'result/figure_before')
    p.add_argument('--after-dir',type=Path,default=PROJECT/'result/figure_after')
    p.add_argument('--from-export',action='store_true',help='Use current exported mapping archives without rebuilding UI data')
    args=p.parse_args()
    if not args.from_export:build(args.data_dir,PROJECT/'replay/web/data',args.result_dir,dataset=args.dataset)
    export(args.result_dir,args.before_dir,args.after_dir)
