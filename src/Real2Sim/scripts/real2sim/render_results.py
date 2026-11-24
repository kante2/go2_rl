#!/usr/bin/env python3
"""Render static figures from saved result arrays and metrics."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'real2sim'

from .config import DEFAULT_DATASET, result_path
from .plot_mapping import plot
from .plot_torque import plot_torque

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default=DEFAULT_DATASET)
    parser.add_argument('--result-dir', type=Path)
    args = parser.parse_args()
    directory = args.result_dir or result_path(args.dataset)
    plot(directory)
    plot_torque(directory)
