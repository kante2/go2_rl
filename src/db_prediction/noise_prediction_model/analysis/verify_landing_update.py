#!/usr/bin/env python3
"""Verify refreshed landing files against source kinematics, PCM and SciPy."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from noise_prediction_model.analysis.landing_event_dynamics import ROOT, REL_MS, DRIVERS, digest, centered_rate
from go2_kinematics import add_foot_kinematics
from contact_event_extraction import continuous_segments
from noise_prediction_model.analysis.summarize_landing_spearman import measured_event_speeds
from noise_prediction_model.paths import resolve_recorded_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path,
        default=ROOT / 'noise_prediction_model/results/analysis/landing_dynamics')
    args = parser.parse_args()
    base = args.input_dir
    metadata = json.loads((base / 'landing_dynamics_metadata.json').read_text())
    events = pd.read_csv(base / 'landing_event_metrics.csv')
    samples = pd.read_csv(base / 'landing_aligned_samples.csv')
    points = pd.read_csv(base / 'trials/spearman_summary.csv')
    assert events.event_id.is_unique and points.event_id.is_unique
    assert set(events.event_id) == set(points.event_id)
    assert events.split.eq('train').all()
    assert len(samples) == len(events) * len(REL_MS)
    assert events.trial.nunique() == metadata['n_trials']
    figure_names = {p.name.removesuffix('_landing_profiles.png') for p in (base / 'trials').glob('*_landing_profiles.png')}
    assert figure_names == set(events.trial)
    matrices = {name: samples[name].to_numpy().reshape(len(events), len(REL_MS))
                for name in (*DRIVERS, 'noise_5ms_dbfs', 'noise_change_db', 'relative_time_ms')}
    np.testing.assert_array_equal(matrices['relative_time_ms'], np.broadcast_to(REL_MS, matrices['relative_time_ms'].shape))
    np.testing.assert_array_equal(samples.event_id.to_numpy().reshape(len(events), len(REL_MS))[:, 0], events.event_id)
    checked_windows = pcm_checks = correlation_checks = 0
    for trial, local in events.groupby('trial', sort=True):
        source = pd.read_csv(ROOT / f'preprocessed/1007/train/{trial}.csv')
        source = add_foot_kinematics(source, velocity_frame=local.body_velocity_frame.iloc[0])
        t = source.time_abs.to_numpy()
        segments, _ = continuous_segments(source)
        with np.load(ROOT / f'data/1007/train/{trial}.npz', allow_pickle=False) as raw:
            audio = raw['audio'].astype(float)
            audio_t = float(raw['audio_t0']) + float(raw['audio_slope']) * np.arange(1, len(audio) + 1)
        audio_t -= float(source.audio_delay_s.iloc[0])
        selected = local.loc[[group.index[len(group)//2] for _, group in local.groupby('leg', sort=True)]]
        for index, event in selected.iterrows():
            center = np.searchsorted(t, event.contact_time_abs - 1e-8)
            indices = center + (REL_MS // 2)
            leg = event.leg.lower()
            force = source[f'ls_ff_{leg}'].to_numpy()
            vz = source[f'foot_velocity_world_{leg}_z_mps'].to_numpy()
            expected = dict(force=force[indices], vz=vz[indices],
                force_rate=centered_rate(force, segments)[indices],
                vz_rate=centered_rate(vz, segments)[indices])
            for name in DRIVERS:
                np.testing.assert_allclose(matrices[name][index], expected[name], rtol=1e-9, atol=1e-6)
            checked_windows += 1
            for pos in (0, 10, 24, 25, 26, 35, 50, 75):
                bounds = [t[indices[pos]] - .0025, t[indices[pos]] + .0025]
                left, right = np.searchsorted(audio_t, bounds, side='left')
                value = 20 * np.log10(max(float(np.std(audio[left:right])), 1e-12) / 32768.)
                np.testing.assert_allclose(matrices['noise_5ms_dbfs'][index, pos], value, atol=2e-6, rtol=0)
                pcm_checks += 1
            for start, stop, field in [(-.050, -.020, 'noise_baseline_rms_dbfs'), (0., .040, 'noise_post40_rms_dbfs')]:
                left, right = np.searchsorted(audio_t, event.contact_time_abs + np.array([start, stop]), side='left')
                value = 20 * np.log10(max(float(np.std(audio[left:right])), 1e-12) / 32768.)
                np.testing.assert_allclose(event[field], value, atol=2e-6, rtol=0)
                pcm_checks += 1
            use = (REL_MS >= -20) & (REL_MS <= 60)
            for name in DRIVERS:
                actual = spearmanr(matrices[name][index, use], matrices['noise_5ms_dbfs'][index, use]).statistic
                np.testing.assert_allclose(actual, event[f'{name}_waveform_spearman_rho'], atol=1e-9, equal_nan=True)
                correlation_checks += 1
        speeds = measured_event_speeds(local, ROOT / f'preprocessed/1007/train/{trial}.csv')
        joined = points[points.trial.eq(trial)].merge(speeds, on='event_id', suffixes=('_saved', '_recomputed'), validate='one_to_one')
        for field in ('measured_vx_median_mps', 'measured_planar_speed_median_mps'):
            np.testing.assert_allclose(joined[field+'_saved'], joined[field+'_recomputed'], atol=1e-10)
    protected = metadata['input_sha256']
    assert all(digest(resolve_recorded_path(path)) == expected for path, expected in protected.items())
    distribution_metadata = json.loads((base / 'noise_distributions/metadata.json').read_text())
    n_distribution_samples = len(events) * 41
    assert distribution_metadata['n_samples'] == n_distribution_samples
    density = pd.read_csv(base / 'noise_distributions/noise_distribution_histogram.csv')
    for _, group in density.groupby('feature'):
        assert int(group['n_samples'].sum()) == n_distribution_samples
    result = dict(passed=True, n_events=len(events), n_trials=int(events.trial.nunique()),
        n_aligned_samples=len(samples), n_distribution_samples=n_distribution_samples,
        kinematics_windows_checked=checked_windows, direct_pcm_windows_checked=pcm_checks,
        scipy_spearman_checked=correlation_checks, all_event_measured_speeds_verified=True,
        protected_input_files=len(protected), all_sources_unchanged=True,
        no_test_used_for_landing_analysis=True, script_sha256=digest(__file__))
    (base / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    (base / 'noise_distributions/verification.json').write_text(json.dumps(dict(
        passed=True, n_events=len(events), n_samples=n_distribution_samples,
        histogram_counts_verified=True, source_verification=str(base / 'verification.json')), indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
