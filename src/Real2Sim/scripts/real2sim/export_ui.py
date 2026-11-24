#!/usr/bin/env python3
"""Serialize saved result arrays and metrics for replay; never read raw logs."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'real2sim'

from .config import DEFAULT_DATASET, RESULT_SCHEMA, UI_INDEX_KEYS, result_path
from .storage import publish_directory


def nullable(values):
    values = np.asarray(values)
    result = np.round(values, 6).astype(object)
    result[~np.isfinite(values)] = None
    return result.tolist()


def trial_document(directory, entry):
    name = entry['id']
    trial = json.loads((directory / 'metrics' / f'{name}.json').read_text())
    with np.load(directory / 'processed' / f'{name}_mapping.npz', allow_pickle=False) as arrays:
        t = np.round(arrays['t'], 6).tolist()
        mask = arrays['steady'].astype(bool).tolist()
        for mode in ('before', 'after'):
            view = trial[mode]
            view.update(t=t, real={}, sim={}, evaluationMask=mask)
            quantities = {}
            for quantity in ('q', 'dq', 'ddq', 'tau'):
                for domain in ('real', 'sim'):
                    values = arrays[f'{domain}_{quantity}_{mode}']
                    if values.shape != (len(t), 12):
                        raise ValueError(f'{name}: inconsistent {domain}/{quantity} shape')
                    view[domain][quantity] = np.round(values, 6).tolist()
                quantities[quantity] = dict(
                    joint=nullable(arrays[f'{mode}_{quantity}_rmse_joint']),
                    combined=nullable(arrays[f'{mode}_{quantity}_rmse_combined']))
            view['errors']['quantities'] = quantities
            view['force']['real_counts'] = np.round(arrays['real_foot_sensor_raw'], 5).tolist()
            view['force']['sim_N'] = np.round(arrays[f'sim_foot_fz_N_{mode}'], 5).tolist()
    return trial


def export_ui(directory, require_complete=True):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('resultSchema') != RESULT_SCHEMA:
        raise ValueError('Unsupported analysis result schema')
    if require_complete and manifest.get('status') != 'complete':
        raise ValueError('Analysis result is not complete')
    index = {key: manifest[key] for key in UI_INDEX_KEYS}
    with tempfile.TemporaryDirectory(prefix='.ui-', dir=directory) as temporary:
        stage = Path(temporary) / 'ui'
        stage.mkdir()
        for entry in index['trials']:
            trial = trial_document(directory, entry)
            (stage / entry['file']).write_text(json.dumps(trial, separators=(',', ':'), allow_nan=False))
        (stage / 'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2, allow_nan=False))
        publish_directory(stage, directory / 'ui')
    return index


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default=DEFAULT_DATASET)
    parser.add_argument('--result-dir', type=Path)
    args = parser.parse_args()
    directory = args.result_dir or result_path(args.dataset)
    index = export_ui(directory)
    print(f'Exported {len(index["trials"])} saved trials to {directory / "ui"}')
