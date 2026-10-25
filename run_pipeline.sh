#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python preprocessing/preprocess.py --label-scope project8 --sample-size 0
python run_experiments.py --suite
python evaluation/plot_results.py --results results/main
