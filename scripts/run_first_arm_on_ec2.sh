#!/usr/bin/env bash
# Execute only on a dedicated EC2 Linux GPU host, from an SSH/SSM session.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
PILOT_CONFIG="${PILOT_CONFIG:-configs/pilot.json}"
if [[ ! "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo 'RUN_ID must contain only letters, numbers, underscores or hyphens.' >&2
  exit 1
fi
PILOT_MODE="${PILOT_MODE:-single}"
case "$PILOT_MODE" in
  single) result_suffix=first-arm; sweep_hours=1 ;;
  matched) result_suffix=matched; sweep_hours=1.6 ;;
  validation) result_suffix=validation; sweep_hours=5.25 ;;
  heldout) result_suffix=heldout; sweep_hours=3.2 ;;
  eviction) result_suffix=eviction; sweep_hours=1.8 ;;
  headroom) result_suffix=headroom; sweep_hours=1.8 ;;
  telemetry) result_suffix=telemetry; sweep_hours=1.8 ;;
  *) echo 'PILOT_MODE must be single, matched, validation, headroom, eviction, heldout or telemetry' >&2; exit 1 ;;
esac
if [[ "$PILOT_MODE" == validation && -n "${PILOT_POLICY:-}${PILOT_CAPACITY:-}${PILOT_SEED:-}" ]]; then
  echo 'Validation mode runs the full frozen matrix; filters are not allowed' >&2; exit 1
fi
python3 -m lab.aws_guard >/dev/null
command -v docker >/dev/null
docker info >/dev/null
WORKLOAD="workloads/$RUN_ID"
RESULTS="results/$RUN_ID-$result_suffix"
if [[ -e "$WORKLOAD" || -e "$RESULTS" ]]; then
  echo 'Run paths already exist. Choose a new RUN_ID; existing artifacts are preserved.' >&2
  exit 1
fi

python3 -m venv .venv
.venv/bin/pip install -r requirements-client.txt
if [[ ! -e deployment/image-lock.json ]]; then
  bash scripts/build_on_ec2.sh
fi
IMAGE_ID=$(python3 -c 'import json; print(json.load(open("deployment/image-lock.json"))["image_id"])')
MODEL=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"])' "$PILOT_CONFIG")
docker image inspect "$IMAGE_ID" >/dev/null
if [[ -z "${MODEL_REVISION:-}" ]]; then
  MODEL_REVISION=$(docker run --rm --entrypoint python3 "$IMAGE_ID" -c \
    'import sys; from huggingface_hub import HfApi; print(HfApi().model_info(sys.argv[1]).sha)' "$MODEL")
fi
if [[ ! "$MODEL_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  echo 'The model revision could not be resolved to an exact commit.' >&2
  exit 1
fi
mkdir -p deployment/model-cache
docker run --rm --entrypoint python3 --network host \
  -v "$PWD:/experiment" -w /experiment \
  -v "$PWD/deployment/model-cache:/root/.cache/huggingface" \
  "$IMAGE_ID" -m lab.workload --config "$PILOT_CONFIG" \
  --output "$WORKLOAD" --model-revision "$MODEL_REVISION"

# Preserve the source/provenance and diagnostics even if the pilot fails.
mkdir -p handoff-results
finish() {
  run_status=$?
  trap - EXIT
  archive_paths=(
    README.md VALIDATION.md MANIFEST.sha256 configs lab scripts vendor \
    deployment/Dockerfile deployment/image-lock.json deployment/ec2-host.json \
    "$WORKLOAD"
  )
  if [[ -d "$RESULTS" ]]; then archive_paths+=("$RESULTS"); fi
  if tar -czf "handoff-results/$RUN_ID.tar.gz" "${archive_paths[@]}"; then
    echo "Run status: $run_status; artifacts: handoff-results/$RUN_ID.tar.gz"
  else
    echo 'Artifact packaging failed. Preserve the workload, results and deployment directories.' >&2
    if [[ "$run_status" == 0 ]]; then run_status=1; fi
  fi
  echo 'EC2 is still running. Stop it through AWS after securing the artifacts.'
  exit "$run_status"
}
trap finish EXIT
arm_args=(--one-arm)
if [[ "$PILOT_MODE" == eviction || "$PILOT_MODE" == heldout ]]; then
  arm_args=(--comparison-screen --trace-all)
elif [[ "$PILOT_MODE" == headroom ]]; then
  arm_args=(--pilot --trace-all)
elif [[ "$PILOT_MODE" == telemetry ]]; then
  arm_args=(--pilot --trace-overhead)
elif [[ "$PILOT_MODE" == validation ]]; then
  arm_args=(--pilot)
elif [[ "$PILOT_MODE" == matched ]]; then
  if [[ -n "${PILOT_POLICY:-}" ]]; then
    echo 'Matched mode cannot select a single policy' >&2; exit 1
  fi
  arm_args=(--capacity "${PILOT_CAPACITY:-constrained}" --seed "${PILOT_SEED:-41}" --pilot)
elif [[ -n "${PILOT_POLICY:-}" ]]; then
  : "${PILOT_CAPACITY:?}" "${PILOT_SEED:?}"
  arm_args=(--policy "$PILOT_POLICY" --capacity "$PILOT_CAPACITY" --seed "$PILOT_SEED" --pilot)
fi
.venv/bin/python -m lab.sweep --workload "$WORKLOAD" \
  --output "$RESULTS" "${arm_args[@]}" --max-hours "$sweep_hours"

if [[ "$PILOT_MODE" == matched || "$PILOT_MODE" == validation || "$PILOT_MODE" == headroom || "$PILOT_MODE" == eviction || "$PILOT_MODE" == heldout ]]; then
  timeout 120 .venv/bin/python -m lab.compare "$RESULTS" --workload "$WORKLOAD"
elif [[ "$PILOT_MODE" == telemetry ]]; then
  timeout 120 .venv/bin/python -m lab.telemetry_report "$RESULTS" --workload "$WORKLOAD"
fi
