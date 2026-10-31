#!/usr/bin/env python3
"""predict_live.py with the v2 comparison dashboard; inference code is unchanged."""
import sys

import web_dashboard_v2

# predict_live imports web_dashboard.Dashboard lazily, after argument parsing.
sys.modules['web_dashboard'] = web_dashboard_v2

import predict_live  # noqa: E402

if __name__ == '__main__':
    predict_live.main()
