#!/usr/bin/env python3
"""Replace synchronization figures from saved matches; keep data and contact PNGs intact."""
from pathlib import Path
import argparse
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/low_noise_matplotlib')
import matplotlib
matplotlib.use('Agg')
from common_time import TrialReader
from time_synchronization import SyncConfig, audio_onsets
from synchronization_figures import plot_trial_figures, plot_overview, load_middle_checks

ROOT = Path(__file__).resolve().parents[1]
KEEP = ('contact_force_velocity_with_noise.png', 'contact_noise.png')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refresh(output):
    figure_dir = output/'figure'
    manifest = json.loads((output/'split.json').read_text())
    protected = {path: digest(path) for path in output.rglob('*')
                 if path.is_file() and (path.suffix != '.png' or path.name in KEEP)}
    summaries = []
    middle = load_middle_checks(figure_dir)
    for item in manifest['raw_files']:
        folder = figure_dir/item['trial']
        summary = json.loads((folder/'synchronization.json').read_text())
        events = json.loads((folder/'footfall_matches.json').read_text())
        trial = TrialReader(item['path'])
        _, audio = audio_onsets(trial, SyncConfig(**summary['config']))
        plot_trial_figures(trial, summary, events, audio, folder, middle.get(trial.name))
        summaries.append(summary)
        print(f"{item['split']}/{trial.name}: summary + detail", flush=True)
    plot_overview(summaries, figure_dir, middle)
    for path, expected in protected.items():
        if digest(path) != expected:
            raise RuntimeError(f'Protected data or contact figure changed: {path}')
    expected_png = {'synchronization.png', 'synchronization_detail.png', *KEEP}
    for item in manifest['raw_files']:
        folder = figure_dir/item['trial']
        actual = {path.name for path in folder.glob('*.png')}
        if actual != expected_png:
            raise RuntimeError(f'Unexpected figures in {folder}: {actual}')
    print(f'Complete: {len(summaries)} sequences, 4 figures each + 1 overview; '
          f'{len(protected)} existing data/contact artifacts unchanged.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'preprocessed/1104')
    refresh(parser.parse_args().output_dir)
