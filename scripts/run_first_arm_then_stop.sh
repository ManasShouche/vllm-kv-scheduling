#!/usr/bin/env bash
# Invoked by the EC2 supervisor as a detached root service. Never run locally.
set -uo pipefail
cd "$(dirname "$0")/.."
python3 -m lab.aws_guard >/dev/null || exit 1
finish() {
  run_status=$?
  trap - EXIT
  echo "First-arm service exited with status $run_status; requesting host shutdown."
  sync
  # The independent deadline remains armed if this immediate request fails.
  shutdown -h now || echo 'Immediate shutdown failed; the host deadline remains armed.' >&2
  exit "$run_status"
}
trap finish EXIT
bash scripts/run_first_arm_on_ec2.sh
exit "$?"
