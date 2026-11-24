"""Shared analysis paths, dataset selection and physical channel names."""
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parents[1]
DEFAULT_DATASET = '1118_RL_base'
LEGS = ('FR', 'FL', 'RR', 'RL')
JOINTS = tuple(f'{leg}_{joint}_joint' for leg in LEGS for joint in ('hip', 'thigh', 'calf'))
RESULT_SCHEMA = 'go2-real2sim-results-v1'
UI_SCHEMA = 'go2-actor-mapping-v4'
UI_INDEX_KEYS = ('schema', 'dataset', 'joints', 'feet', 'units', 'trials', 'excludedTrials')


def validate_dataset(dataset):
    if not dataset or Path(dataset).name != dataset or dataset in ('.', '..'):
        raise ValueError('dataset must be a single folder name')
    return dataset


def result_path(dataset=DEFAULT_DATASET):
    return PROJECT / 'result' / validate_dataset(dataset)
