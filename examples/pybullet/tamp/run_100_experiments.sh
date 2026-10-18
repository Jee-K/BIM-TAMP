#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${BIM_TAMP_PYTHON:-python3}"
exec "$python_bin" -u "$script_dir/run_100_experiments.py" "$@"
