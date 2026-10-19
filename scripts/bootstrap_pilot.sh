#!/usr/bin/env bash
# Invoked by EC2 user-data on a new dedicated host. Launch controller must verify
# instance-initiated shutdown=stop. Uses an existing SSM/S3 role, no IAM changes.
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
: "${SOURCE_S3_URI:?}" "${SOURCE_SHA256:?}" "${RESULT_S3_URI:?}" "${RUN_ID:?}"
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]]
[[ "$SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]]
case "${PILOT_MODE:-single}" in
  single) host_minutes=110; job_minutes=95 ;;
  matched) host_minutes=140; job_minutes=125 ;;
  validation) host_minutes=360; job_minutes=345 ;;
  heldout) host_minutes=240; job_minutes=225 ;;
  eviction) host_minutes=140; job_minutes=125 ;;
  headroom) host_minutes=140; job_minutes=125 ;;
  telemetry) host_minutes=140; job_minutes=125 ;;
  *) echo 'Invalid PILOT_MODE' >&2; exit 1 ;;
esac
[[ "$(uname -s)" == Linux && "$EUID" == 0 ]]
# Confirm EC2 metadata is reachable before arming a host shutdown.
token=$(curl --noproxy '*' -fsS --max-time 5 -X PUT \
  -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' http://169.254.169.254/latest/api/token)
curl --noproxy '*' -fsS --max-time 5 -H "X-aws-ec2-metadata-token: $token" \
  http://169.254.169.254/latest/dynamic/instance-identity/document >/dev/null
unset token
mkdir -p /opt/kv-pressure-run
exec > >(tee -a /var/log/kv-pressure-bootstrap.log) 2>&1

finish() {
  status=$?
  trap - EXIT
  set +e
  printf '%s\n' "$status" > /opt/kv-pressure-run/exit-code.txt
  # The outer timeout can interrupt Python before its finally block completes.
  timeout 30 docker logs kv-pressure-lab-server > /opt/kv-pressure-run/server-recovery.log 2>&1
  cp /var/log/kv-pressure-bootstrap.log /opt/kv-pressure-run/bootstrap.log
  # Package startup failures too; model caches and dependencies are reproducible.
  timeout 180 tar -czf /opt/kv-pressure-results.tar.gz \
    --exclude='.venv' --exclude='model-cache' --exclude='__pycache__' \
    -C /opt kv-pressure-run
  packaged=$?
  if [[ "$packaged" == 0 ]]; then
    cd /opt
    sha256sum kv-pressure-results.tar.gz > kv-pressure-results.tar.gz.sha256
    timeout 180 aws s3 cp kv-pressure-results.tar.gz "$RESULT_S3_URI/results.tar.gz" --sse AES256 --only-show-errors
    uploaded=$?
    if [[ "$uploaded" == 0 ]]; then
      timeout 60 aws s3 cp kv-pressure-results.tar.gz.sha256 "$RESULT_S3_URI/results.tar.gz.sha256" --sse AES256 --only-show-errors
    fi
  fi
  sync
  shutdown -h now
  exit "$status"
}
trap finish EXIT
# Deadline covers setup/downloads as well as inference. It is independent of
# the client connection. Outer workload timeout leaves time for result upload.
systemd-run --unit="kv-pressure-boot-deadline-$RUN_ID" --on-active="${host_minutes}m" \
  --timer-property=AccuracySec=1s -- /usr/sbin/shutdown -h now
systemctl is-active --quiet "kv-pressure-boot-deadline-$RUN_ID.timer"
aws s3 cp "$SOURCE_S3_URI" /opt/kv-pressure-source.tar.gz --only-show-errors
printf '%s  %s\n' "$SOURCE_SHA256" /opt/kv-pressure-source.tar.gz | sha256sum -c -
tar -xzf /opt/kv-pressure-source.tar.gz -C /opt/kv-pressure-run
cd /opt/kv-pressure-run/kv-pressure-lab
sha256sum -c MANIFEST.sha256
nvidia-smi
docker info >/dev/null
export RUN_ID
timeout --signal=TERM --kill-after=30s "${job_minutes}m" bash scripts/run_first_arm_on_ec2.sh
