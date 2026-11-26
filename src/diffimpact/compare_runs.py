#!/usr/bin/env python3
"""여러 DiffImpact 실행을 같은 검증 세트(실행 간 교집합)에서 비교.

창 길이가 달라도 비교할 수 있는 지표만 쓴다 (모두 세트 첫 착지 기준, 실현값 샘플 8회 평균):
  - 착지 후 0~40 ms 레벨 MAE / 편향 / 상관
  - 착지 후 0~80 ms 대역별 레벨 MAE (50-200, 200-1000, 1000-4000, 4000-12000 Hz)
  - 시뮬 착지에 적용했을 때 녹화별 중앙값 레벨과 실측의 상관·편향
출력: runs/compare/<이름>.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_diffimpact as T          # noqa: E402
from diffimpact_model import DiffImpact  # noqa: E402


def run_eval(run, dev, ckpt='model_feat.pt'):
    ck = torch.load(run / ckpt, weights_only=False)
    data = T.run_data_dir(run)
    T.FEATURES = list(ck['features'])
    model = DiffImpact(**ck['config']).to(dev); T.load_state(model, ck['state']); model.eval()
    out = {}
    for src in ('real', 'sim'):
        d = T.load_pairs(data / f'pairs_{src}.npz')
        d['feats'] = ((d['feats'] - ck['feat_mu']) / ck['feat_sd']).astype(np.float32)
        with np.load(data / f'pairs_{src}.npz') as z:
            keys = list(zip(z['dataset'].astype(str), z['trial'].astype(str), z['group'].astype(int)))
        ev = T.evaluate(model, d, dev, use_feats=True, draws=8)
        res = dict(keys=keys, pred40=T.predicted_post40(model, ev['samples']), dataset=d['dataset'], trial=d['trial'])
        if src == 'real':
            ns = argparse.Namespace(pre=d['pre'], fs=d['fs'])
            res['real40'] = T.dbfs_window(d['audio'], ns)
            res['band_pred'] = np.mean([T.band_db(model, x, dev) for x in ev['samples']], 0)
            res['band_real'] = T.band_db(model, d['audio'], dev)
        out[src] = res
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--name', default='runs')
    p.add_argument('--ckpt', default='model_feat.pt', help='각 실행에서 읽을 체크포인트 (예: model_feat_cal.pt)')
    a = p.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    R = {r.name: run_eval(r, dev, a.ckpt) for r in a.runs}
    val_keys = set.intersection(*[{k for k in v['real']['keys'] if (k[0], k[1]) in T.VAL_TRIALS} for v in R.values()])
    sim_keys = set.intersection(*[set(v['sim']['keys']) for v in R.values()])
    rows = []
    bname = [f'{lo}-{hi}Hz' for lo, hi in T.BANDS]
    for name, v in R.items():
        idx = [i for i, k in enumerate(v['real']['keys']) if k in val_keys]
        order = np.argsort([str(v['real']['keys'][i]) for i in idx]); idx = np.array(idx)[order]
        pr, rl = v['real']['pred40'][idx], v['real']['real40'][idx]
        row = dict(run=name, n_val=len(idx), post40_mae_db=np.abs(pr - rl).mean(), post40_bias_db=(pr - rl).mean(),
                   post40_pearson=np.corrcoef(pr, rl)[0, 1])
        bmae = np.abs(v['real']['band_pred'][idx] - v['real']['band_real'][idx]).mean(0)
        row.update({f'band80_mae_{b}': x for b, x in zip(bname, bmae)})
        # 시뮬 적용: 녹화별 중앙값 (같은 시뮬 세트 교집합) vs 실측 녹화별 중앙값 (전체 실측 세트)
        s = v['sim']; si = np.array([i for i, k in enumerate(s['keys']) if k in sim_keys])
        recs = sorted(set(zip(s['dataset'][si], s['trial'][si])))
        sim_med, real_med = [], []
        for ds, tr in recs:
            ms = si[(s['dataset'][si] == ds) & (s['trial'][si] == tr)]
            mr = (v['real']['dataset'] == ds) & (v['real']['trial'] == tr)
            sim_med.append(np.median(s['pred40'][ms])); real_med.append(np.median(v['real']['real40'][mr]))
        sim_med, real_med = np.array(sim_med), np.array(real_med)
        row.update(sim_rec_pearson=np.corrcoef(sim_med, real_med)[0, 1], sim_rec_bias_db=np.median(sim_med - real_med))
        rows.append(row)
    df = pd.DataFrame(rows)
    (HERE / 'runs/compare').mkdir(parents=True, exist_ok=True)
    df.to_csv(HERE / f'runs/compare/{a.name}.csv', index=False, float_format='%.3f')
    pd.set_option('display.width', 250)
    print(f'공통 검증 세트 {len(val_keys)}개, 공통 시뮬 세트 {len(sim_keys)}개')
    print(df.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
