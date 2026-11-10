#!/usr/bin/env python3
"""Verify saved synchronization, source hashes, PCM samples and split aggregation."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from synchronization_diagnostics import write_json

ROOT = Path(__file__).resolve().parents[1]


def verify(output):
    manifest = json.loads((output/'split.json').read_text())
    config = manifest['config']
    raw_files = manifest['raw_files']
    assert not set(manifest['train_trials']) & set(manifest['test_trials'])
    assert not any(item['trial'].startswith('no_vel') for item in raw_files)
    summaries = {item['trial']: json.loads((output/'figure'/item['trial']/'synchronization.json').read_text())
                 for item in raw_files}
    sample_checks = 0
    for split in ('train', 'test'):
        quality = pd.read_csv(output/split/'quality.csv')
        assert quality.trial.is_unique
        aggregate = pd.read_csv(output/split/'all_trials.csv', usecols=['trial', 'synchronization_passed'])
        eligible = set(quality.loc[quality.included_in_all_trials, 'trial'])
        assert set(aggregate.trial) == eligible
        assert aggregate.synchronization_passed.all()
        counts = aggregate.trial.value_counts()
        for row in quality.to_dict('records'):
            assert (row['trial'] in eligible) == (row['status'] == 'pass')
            if row['trial'] in eligible:
                assert counts[row['trial']] == row['selected_rows']
    for item in raw_files:
        path = Path(item['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], f'Raw changed: {path}'
        trial = item['trial']; split = item['split']; summary = summaries[trial]
        frame = pd.read_csv(output/split/f'{trial}.csv', float_precision='round_trip')
        assert len(frame) > 0 and np.all(np.diff(frame.time_abs) > 0)
        delay = summary['applied_audio_delay_s']
        np.testing.assert_allclose(frame.audio_delay_s, delay, rtol=0, atol=1e-12)
        assert set(frame.synchronization_status) == {summary['status']}
        assert (frame.synchronization_passed == summary['passed']).all()
        start_ids = set(summary['calibration']['inlier_group_ids'])
        end_ids = set(summary['validation']['inlier_group_ids'])
        assert not start_ids & end_ids
        if summary['passed']:
            assert summary['calibration']['usable'] and summary['validation']['usable']
            assert abs(summary['end_residual_ms']) <= config['tolerance_s']*1000+1e-6
            assert summary['end_abs_error_p90_ms'] <= config['tolerance_s']*1000+1e-6
        if not summary['correction_accepted']:
            assert delay == 0
        selected = frame.iloc[np.linspace(0, len(frame)-1, min(25, len(frame)), dtype=int)]
        metadata = json.loads((output/split/'metadata.json').read_text())
        assert metadata['robot_interpolation'] == 'previous', 'Verifier expects previous interpolation'
        hz = metadata['target_hz']
        with np.load(path, allow_pickle=False) as raw:
            rt = raw['ls_t']
            indices = np.searchsorted(rt, selected.time_abs.to_numpy(), side='right')-1
            np.testing.assert_allclose(selected[[f'ls_ff_{leg}' for leg in ('fr','fl','rr','rl')]], raw['ls_ff'][indices])
            for col, index in [('ls_q_fr_hip', 0), ('ls_dq_rl_calf', 11), ('ls_tau_fl_thigh', 4)]:
                key = 'ls_'+col.split('_')[1]
                np.testing.assert_allclose(selected[col], raw[key][indices, index], atol=1e-12)
            pcm = raw['audio'].astype(float)
            slope = float(raw['audio_slope']); t0 = float(raw['audio_t0'])
            for row in selected.to_dict('records'):
                # Recompute raw PCM std for the two original feature grid points.
                at = row['time_abs']+delay
                lower = np.floor(at*hz)/hz
                upper = lower+1/hz
                values = []
                for center in (lower, upper):
                    left = int(np.ceil((center-.010-t0)/slope-1))
                    right = int(np.ceil((center+.010-t0)/slope-1))
                    assert 0 <= left < right <= len(pcm)
                    values.append(float(np.std(pcm[left:right])))
                weight = (at-lower)*hz
                expected = values[0]+weight*(values[1]-values[0])
                np.testing.assert_allclose(row['audio_rms_20ms'], expected, atol=1e-4, rtol=1e-5)
                sample_checks += 1
    result = dict(passed=True, raw_hashes_checked=len(raw_files), pcm_and_sensor_rows_checked=sample_checks,
                  trials=len(raw_files), synchronized=sum(s['passed'] for s in summaries.values()),
                  review=sum(not s['passed'] for s in summaries.values()), split_overlap=False)
    write_json(output/'figure/synchronization_verification.json', result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'preprocessed/1104')
    verify(parser.parse_args().output_dir)
