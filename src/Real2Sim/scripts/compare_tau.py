#!/usr/bin/env python3
"""Plot torque from an existing result; run_converter.sh performs analysis."""
import argparse
from pathlib import Path
from real2sim.config import DEFAULT_DATASET, result_path
from real2sim.plot_torque import plot_torque

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default=DEFAULT_DATASET)
    parser.add_argument('--result-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--from-export', action='store_true', help='Compatibility flag; this tool always uses saved results')
    args = parser.parse_args()
    plot_torque(args.result_dir or result_path(args.dataset), args.output_dir)
