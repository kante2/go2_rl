#!/usr/bin/env python3
"""Preprocess split datasets with start-footfall calibration and held-out end checks.

Default: data/1104/{train,test} -> preprocessed/1104/{train,test,figure}.
no_vel* inputs are excluded. Per-trial CSVs preserve review cases with explicit
status; all_trials.csv includes only synchronized, sufficiently long trials.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/low_noise_matplotlib')
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from common_time import TrialReader, create_common_time
from time_synchronization import SyncConfig, estimate_footfall_delay, apply_audio_delay
from steady_state import detect_steady_state
from dataset_output import csv_columns, iter_rows, write_csv
from synchronization_diagnostics import write_json, save_trial_diagnostics, save_run_summary
from contact_event_extraction import analyze_frame, save_trial_result, collect_event_tables
from contact_noise_visualization import save_contact_summary

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT/'data/1104'
OUTPUT_DIR = PROJECT_ROOT/'preprocessed/1104'
FILE_NAME = '0.2_1'


def configured_trial_path():
    path = DATA_DIR/'train'/f'{FILE_NAME}.npz'
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def discover_split_inputs(root):
    sources = {split: sorted(p for p in (root/split).glob('*.npz')
                              if not p.stem.startswith('no_vel')) for split in ('train', 'test')}
    if not all(sources.values()):
        raise ValueError('nonempty train/ and test/ NPZ directories required')
    train_ids = {p.stem for p in sources['train']}
    if train_ids & {p.stem for p in sources['test']}:
        raise ValueError('trial IDs overlap across train/test')
    return sources


def build_rows(trial, corrected, state, selection, summary):
    columns = csv_columns()+['synchronization_status', 'synchronization_passed']
    rows = [row+[summary['status'], summary['passed']]
            for row in iter_rows(trial, corrected, state, selection)]
    return pd.DataFrame(rows, columns=columns)


def run(args):
    sources = discover_split_inputs(args.data_dir)
    figure_dir = args.figure_dir or args.output_dir/'figure'
    figure_dir.mkdir(parents=True, exist_ok=True)
    config = SyncConfig(tolerance_s=args.sync_tolerance_ms/1000)
    summaries, contact_summary, manifest = [], [], []
    for split, paths in sources.items():
        destination = args.output_dir/split
        destination.mkdir(parents=True, exist_ok=True)
        # Rebuild aggregates from this run; never append stale or duplicated rows.
        combined_path = destination/'all_trials.csv'
        temporary = combined_path.with_suffix('.csv.tmp')
        quality = []
        with temporary.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(csv_columns()+['synchronization_status', 'synchronization_passed'])
            for path in paths:
                trial = TrialReader(path)
                print(f'{split}/{trial.name}: estimating footfall synchronization', flush=True)
                summary, matches, audio = estimate_footfall_delay(trial, config)
                summary['split'] = split
                common = create_common_time(trial, args.hz, args.max_gap,
                                            robot_interpolation=args.robot_interpolation)
                corrected = apply_audio_delay(common, summary['applied_audio_delay_s'])
                state = detect_steady_state(corrected)
                frame = build_rows(trial, corrected, state, args.selection, summary)
                duration = len(frame)/args.hz
                eligible = summary['passed'] and duration >= args.min_output
                quality.append(dict(trial=trial.name, status='pass' if eligible else
                                    summary['status'] if not summary['passed'] else 'INSUFFICIENT_DURATION',
                                    synchronization_status=summary['status'], included_in_all_trials=eligible,
                                    selected_rows=len(frame), selected_duration_s=duration,
                                    valid_fraction=float(np.mean(corrected['valid'])),
                                    audio_delay_s=summary['applied_audio_delay_s']))
                folder = figure_dir/trial.name
                save_trial_diagnostics(trial, common, summary, matches, audio, folder, not args.no_plot)
                write_csv(destination/f'{trial.name}.csv', list(frame.columns), frame.itertuples(index=False, name=None))
                if eligible:
                    writer.writerows(frame.itertuples(index=False, name=None))
                if len(frame) >= 2:
                    result = analyze_frame(frame, trial.name, raw_path=path, split=split, sync_summary=summary)
                    save_trial_result(result, trial.name, folder, 10., 5., 6., make_plot=not args.no_plot)
                    contact_summary.extend(result['summary'])
                summaries.append(summary)
                manifest.append(dict(split=split, trial=trial.name, path=str(path.resolve()),
                                     sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                print(f"{split}/{trial.name}: {summary['status']}, delay={summary['applied_audio_delay_s']*1000:+.2f} ms, "
                      f"end residual={summary['end_residual_ms']} ms, rows={len(frame)}", flush=True)
        temporary.replace(combined_path)
        pd.DataFrame(quality).to_csv(destination/'quality.csv', index=False)
        selected = [s for s in summaries if s['split'] == split]
        write_json(destination/'metadata.json', dict(schema_version=3, target_hz=args.hz,
                   max_gap_s=args.max_gap, robot_interpolation=args.robot_interpolation,
                   selection=args.selection, audio_rms_window_s=.020, audio_peak_window_s=1/args.hz,
                   audio_delay_policy='per_sequence_start_footfalls_end_held_out',
                   all_trials_policy='only synchronization PASS and sufficient duration',
                   per_trial_audio_delay_s={s['trial']: s['applied_audio_delay_s'] for s in selected},
                   synchronization_status={s['trial']: s['status'] for s in selected},
                   synchronization_config=selected[0]['config'],
                   trials=[p.stem for p in paths], synchronized_trials=[s['trial'] for s in selected if s['passed']],
                   input_dir=str((args.data_dir/split).resolve())))
    save_contact_summary(figure_dir, contact_summary)
    collect_event_tables(args.output_dir, figure_dir)
    run_info = dict(input_dir=str(args.data_dir.resolve()), output_dir=str(args.output_dir.resolve()),
                    excluded_pattern='no_vel*', split_policy='preserved original train/test',
                    audio_delay_policy='per_sequence_start_footfalls_end_held_out',
                    config=summaries[0]['config'], raw_files=manifest)
    write_json(args.output_dir/'split.json', dict(run_info,
               train_trials=[p.stem for p in sources['train']], test_trials=[p.stem for p in sources['test']],
               train_input=str((args.output_dir/'train/all_trials.csv').resolve()),
               test_input=str((args.output_dir/'test/all_trials.csv').resolve())))
    save_run_summary(figure_dir, summaries, run_info, plots=not args.no_plot)
    passed = sum(s['passed'] for s in summaries)
    print(f'Complete: {passed}/{len(summaries)} synchronized; review cases retained in individual CSVs. {args.output_dir}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--figure-dir', type=Path)
    parser.add_argument('--all', action='store_true', help='all split inputs are processed by default')
    parser.add_argument('--hz', type=float, default=500.)
    parser.add_argument('--max-gap', type=float, default=.020)
    parser.add_argument('--robot-interpolation', choices=['previous', 'linear'], default='previous')
    parser.add_argument('--selection', choices=['walking', 'steady', 'full'], default='walking')
    parser.add_argument('--min-output', type=float, default=5.)
    parser.add_argument('--sync-tolerance-ms', type=float, default=10.)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    if any(not np.isfinite(v) or v <= 0 for v in (args.hz, args.max_gap, args.min_output, args.sync_tolerance_ms)):
        parser.error('rates, gaps, durations and tolerance must be positive and finite')
    if args.data_dir.resolve() == args.output_dir.resolve():
        parser.error('output directory must differ from raw input')
    run(args)


if __name__ == '__main__':
    main()
