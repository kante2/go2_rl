#!/usr/bin/env python3
"""트롯 대각 쌍(FR+RL / FL+RR)을 한 세트로 묶은 DiffImpact 학습 데이터.

입력 : data/<touchdown_*>/touchdowns_{real,sim}.csv, impacts_{real,sim}.npz  (extract_touchdowns.py 결과)
출력 : data/<touchdown_*>/pairs_{real,sim}.csv   세트 1행 (앞발·뒷발 특징을 한 행에)
       data/<touchdown_*>/pairs_{real,sim}.npz   세트 1창: 소리(실로봇) + 두 발의 힘·속도·위치

세트 조건: 한 묶음(80 ms 이내) 안에 대각 두 발이 각각 정확히 한 번씩 착지.
           1발만 있는 묶음, 3발 묶음, 대각이 아닌 조합은 제외하고 개수를 summary에 남긴다.
창 기준  : 세트의 첫 착지 = 0 ms (impacts 와 동일, -50 ~ +250 ms).
다리 순서: 세트 안에서 [앞발, 뒷발]  (FR+RL → [FR, RL], FL+RR → [FL, RR])
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LEGS = ('FR', 'FL', 'RR', 'RL')
DIAGONALS = {'FR+RL': ('FR', 'RL'), 'FL+RR': ('FL', 'RR')}   # (앞발, 뒷발)
LEG_FEATURES = ['t_abs', 'offset_in_group_ms', 'force_air', 'force_at_td', 'force_peak30',
                'force_peak30_minus_air', 'time_to_peak_ms', 'force_rate_max',
                'foot_vx_pre', 'foot_vy_pre', 'foot_vz_pre', 'foot_x', 'foot_y', 'foot_z',
                'swing_s', 'stance_s', 'censored',
                'grf_fz_pre', 'grf_fz_at_td', 'grf_fz_peak20', 'grf_fz_rise20', 'grf_fz_impulse20', 'grf_fz_rate10']
SIM_FEATURES = ['sim_fx_peak', 'sim_fy_peak', 'sim_fz_peak', 'sim_foot_vz_gt_pre', 'sim_body_vz_pre',
                'sim_gt_fz_peak20', 'sim_gt_fz_rise20', 'sim_gt_fz_impulse20', 'sim_gt_fz_rate10']
KEY = ['dataset', 'trial', 'group']


def diagonal_of(legs):
    for name, pair in DIAGONALS.items():
        if sorted(legs) == sorted(pair):
            return name
    return None


def pair_table(td):
    rows, dropped = [], {}
    for key, g in td.groupby(KEY, sort=False):
        name = diagonal_of(list(g.leg))
        if name is None or len(g) != 2:
            reason = f'size{len(g)}' if len(g) != 2 else 'non_diagonal'
            dropped[reason] = dropped.get(reason, 0) + 1
            continue
        front_leg, rear_leg = DIAGONALS[name]
        f = g[g.leg == front_leg].iloc[0]
        r = g[g.leg == rear_leg].iloc[0]
        first = f if f.t_abs <= r.t_abs else r
        row = dict(zip(KEY, key))
        row.update(walk=f.walk, diagonal=name, front_leg=front_leg, rear_leg=rear_leg,
                   lead='front' if first is f else 'rear', t0_abs=first.t_abs, t0_rel=first.t_rel,
                   pair_spread_ms=abs(f.t_abs - r.t_abs) * 1e3,
                   prev_pair_gap_ms=f.prev_group_gap_ms, next_pair_gap_ms=f.next_group_gap_ms,
                   in_impacts_npz=bool(f.in_impacts_npz))
        for c in ('cmd_vx', 'cmd_vy', 'cmd_wz', 'cmd_vx_at_td', 'cmd_vy_at_td', 'cmd_wz_at_td'):
            if c in g:
                row[c] = g[c].mean()
        for prefix, leg in (('front', f), ('rear', r)):
            for c in LEG_FEATURES + [c for c in SIM_FEATURES if c in g]:
                row[f'{prefix}_{c}'] = leg[c]
        # 세트 단위 요약 (DiffImpact 충격 크기 회귀용 후보)
        row['pair_force_peak_sum'] = f.force_peak30_minus_air + r.force_peak30_minus_air
        row['pair_foot_vz_pre_mean'] = (f.foot_vz_pre + r.foot_vz_pre) / 2
        row['pair_foot_vz_pre_min'] = min(f.foot_vz_pre, r.foot_vz_pre)
        rows.append(row)
    return pd.DataFrame(rows), dropped


def pair_arrays(npz, table):
    """impacts npz 에서 세트에 해당하는 창만 골라 두 발 [앞, 뒤] 로 줄인다."""
    keys = list(zip(npz['dataset'].astype(str), npz['trial'].astype(str), npz['group'].astype(int)))
    index = {k: i for i, k in enumerate(keys)}
    sel, legs_idx, keep = [], [], []
    for j, r in enumerate(table.itertuples()):
        i = index.get((r.dataset, r.trial, int(r.group)))
        if i is None:
            continue
        sel.append(i); keep.append(j)
        legs_idx.append([LEGS.index(r.front_leg), LEGS.index(r.rear_leg)])
    sel = np.array(sel); li = np.array(legs_idx)
    rows = np.arange(len(sel))[:, None]
    out = dict(
        force=npz['force'][sel][rows, :, li].transpose(0, 2, 1),              # (G,T,2)
        foot_vel=npz['foot_vel'][sel][rows, :, li].transpose(0, 2, 1, 3),     # (G,T,2,3)
        foot_pos=npz['foot_pos'][sel][rows, :, li].transpose(0, 2, 1, 3),
        grf=npz['grf'][sel][rows, :, li].transpose(0, 2, 1, 3),               # 토크 기반 접촉력 [N]
        td_offset_ms=npz['td_offset_ms'][sel][rows, li],                      # (G,2)
        leg_index=li.astype(np.int8),
        pair_legs=np.array([[LEGS[a], LEGS[b]] for a, b in li]),
    )
    if 'force_vec' in npz:
        out['force_vec'] = npz['force_vec'][sel][rows, :, li].transpose(0, 2, 1, 3)
    for k in ('audio', 'audio_pre_dbfs', 'audio_post40_dbfs', 'cmd', 'cmd_at_td',
              'trial', 'dataset', 'walk', 'group', 't0_abs', 't0_rel'):
        if k in npz:
            out[k] = npz[k][sel]
    out['diagonal'] = table.diagonal.to_numpy()[keep].astype(str)
    out['lead'] = table.lead.to_numpy()[keep].astype(str)
    out['t_rel_ms'] = npz['t_rel_ms']
    if 'fs' in npz:
        out['fs'] = npz['fs']
    return out, keep


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dir', type=Path, default=HERE / 'data' / 'touchdown_w100_200hz')
    a = p.parse_args()
    summary = {}
    for source in ('real', 'sim'):
        td = pd.read_csv(a.dir / f'touchdowns_{source}.csv')
        table, dropped = pair_table(td)
        with np.load(a.dir / f'impacts_{source}.npz', allow_pickle=False) as z:
            npz = {k: z[k] for k in z.files}
        arrays, keep = pair_arrays(npz, table)
        table['in_pairs_npz'] = False
        table.loc[table.index[keep], 'in_pairs_npz'] = True
        table.to_csv(a.dir / f'pairs_{source}.csv', index=False, float_format='%.10g')
        arrays['meta'] = np.array(json.dumps(dict(
            leg_order='[front, rear] of the diagonal pair', diagonals=DIAGONALS,
            window='t_rel_ms, 0 = first touchdown of the pair', source=f'impacts_{source}.npz')))
        np.savez_compressed(a.dir / f'pairs_{source}.npz', **arrays)
        by = table[table.in_pairs_npz].groupby(['dataset', 'trial']).size()
        summary[source] = dict(
            groups_total=int(td.groupby(KEY).ngroups), pairs=len(table), pairs_in_npz=len(keep),
            dropped_groups=dropped,
            diagonal_counts=table.diagonal.value_counts().to_dict(),
            lead_counts=table.lead.value_counts().to_dict(),
            pair_spread_ms_by_walk=table.groupby('walk').pair_spread_ms.describe(
                percentiles=[.5, .95]).round(1).to_dict('index'),
            pairs_per_trial={f'{d}/{t}': int(n) for (d, t), n in by.items()})
        print(f"{source}: groups {summary[source]['groups_total']} -> pairs {len(table)} "
              f"(npz {len(keep)}), dropped {dropped}, diag {summary[source]['diagonal_counts']}, "
              f"lead {summary[source]['lead_counts']}")
    (a.dir / 'pairs_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
