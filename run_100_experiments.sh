#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$root/examples/pybullet/tamp/run_100_experiments.sh" "$@"
