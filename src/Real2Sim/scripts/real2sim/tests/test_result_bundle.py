"""Result publication and UI export are independent from original recordings."""
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from real2sim.config import result_path
from real2sim.export_ui import export_ui
from real2sim import pipeline, storage


def one_saved_trial(tmp_path):
    source = result_path()
    manifest = json.loads((source / 'manifest.json').read_text())
    entry = dict(manifest['trials'][0], realPath='/missing/raw/real.npz', simPath='/missing/raw/sim.npz')
    manifest['trials'] = [entry]
    bundle = tmp_path / 'result'
    for name in ('processed', 'metrics'):
        (bundle / name).mkdir(parents=True)
    name = entry['id']
    shutil.copy2(source / 'processed' / f'{name}_mapping.npz', bundle / 'processed')
    metadata = json.loads((source / 'metrics' / f'{name}.json').read_text())
    metadata['provenance'] = entry
    (bundle / 'metrics' / f'{name}.json').write_text(json.dumps(metadata))
    (bundle / 'manifest.json').write_text(json.dumps(manifest))
    return bundle, name


def test_ui_export_reads_saved_arrays_and_metrics_without_analysis(tmp_path, monkeypatch):
    bundle, name = one_saved_trial(tmp_path)
    for module in ('pipeline', 'preprocessing', 'time_alignment', 'metrics'):
        monkeypatch.setitem(sys.modules, 'real2sim.' + module, None)
    export_ui(bundle)
    trial = json.loads((bundle / 'ui' / f'{name}.json').read_text())
    metadata = json.loads((bundle / 'metrics' / f'{name}.json').read_text())
    with np.load(bundle / 'processed' / f'{name}_mapping.npz') as arrays:
        for mode in ('before', 'after'):
            assert trial[mode]['metrics'] == metadata[mode]['metrics']
            for domain in ('real', 'sim'):
                for quantity in ('q', 'dq', 'ddq', 'tau'):
                    assert trial[mode][domain][quantity] == np.round(arrays[f'{domain}_{quantity}_{mode}'], 6).tolist()


def test_failed_export_preserves_previous_ui(tmp_path):
    bundle, name = one_saved_trial(tmp_path)
    export_ui(bundle)
    previous = {p.name: p.read_bytes() for p in (bundle / 'ui').glob('*.json')}
    (bundle / 'processed' / f'{name}_mapping.npz').unlink()
    with pytest.raises(FileNotFoundError):
        export_ui(bundle)
    assert previous == {p.name: p.read_bytes() for p in (bundle / 'ui').glob('*.json')}


def test_failure_after_first_analyzed_trial_keeps_published_result(tmp_path, monkeypatch):
    destination = tmp_path / 'published'
    destination.mkdir()
    (destination / 'manifest.json').write_text('previous complete result')
    real, sim = pipeline.paired_paths(pipeline.PROJECT / 'data')[0]
    monkeypatch.setattr(pipeline, 'paired_paths', lambda *_: [(real, sim), (tmp_path / 'missing.npz', sim)])
    with pytest.raises(FileNotFoundError):
        pipeline.build(result_dir=destination, plots=False)
    assert (destination / 'manifest.json').read_text() == 'previous complete result'
    assert list(destination.iterdir()) == [destination / 'manifest.json']
    assert not list(tmp_path.glob('.*-build-*'))


def test_publish_restores_previous_directory_when_switch_fails(tmp_path, monkeypatch):
    stage, destination = tmp_path / 'stage', tmp_path / 'published'
    stage.mkdir(); destination.mkdir()
    (stage / 'manifest.json').write_text('new')
    (destination / 'manifest.json').write_text('old')
    replace = storage.os.replace
    def fail_new_result(source, target):
        if Path(source) == stage:
            raise OSError('simulated publish failure')
        return replace(source, target)
    monkeypatch.setattr(storage.os, 'replace', fail_new_result)
    with pytest.raises(OSError, match='simulated publish failure'):
        storage.publish_directory(stage, destination)
    assert (destination / 'manifest.json').read_text() == 'old'
