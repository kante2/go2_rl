#!/usr/bin/env python3
"""이미 학습된 실행에 레벨 보정(train_diffimpact.calibrate_level)을 적용.
model_feat.pt 는 건드리지 않고 model_feat_cal.pt 로 저장한다."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_diffimpact as T          # noqa: E402
from diffimpact_model import DiffImpact  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('runs', nargs='+', type=Path)
a = p.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
for run in a.runs:
    ck = torch.load(run / 'model_feat.pt', weights_only=False)
    data = T.run_data_dir(run)
    T.FEATURES = list(ck['features'])
    model = DiffImpact(**ck['config']).to(dev); T.load_state(model, ck['state']); model.eval()
    real = T.load_pairs(data / 'pairs_real.npz')
    train = T.subset(real, np.array([(d, t) not in T.VAL_TRIALS for d, t in zip(real['dataset'], real['trial'])]))
    train['feats'] = ((train['feats'] - ck['feat_mu']) / ck['feat_sd']).astype(np.float32)
    cal = T.calibrate_level(model, train, dev)
    torch.save(dict(ck, state=model.state_dict(), level_calibration=cal), run / 'model_feat_cal.pt')
    print(run.name, {k: round(v, 3) for k, v in cal.items()}, flush=True)
