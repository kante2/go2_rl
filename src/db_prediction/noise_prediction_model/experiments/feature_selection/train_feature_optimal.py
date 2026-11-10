#!/usr/bin/env python3
"""Whole-candidate model-based selection and nested group-CV K tuning.

Uses train_op.py's shared training pipeline with an ExtraTrees selector.
The existing standard model and Feature_extraction outputs are preserved.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from threadpoolctl import threadpool_limits
from noise_prediction_model.experiments.feature_selection.model_evaluation import EVALUATION_DIR
from train_op import main

if __name__=='__main__':
    sys.argv[1:1]=['--output',str(EVALUATION_DIR/'feature_optimal'),
        '--selection-method','extra-trees','--k-values','5','10','15','20','30','40','60','86']
    with threadpool_limits(limits=1): main()
