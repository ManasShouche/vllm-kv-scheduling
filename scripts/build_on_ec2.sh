#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m lab.aws_guard > deployment/ec2-host.json
command -v docker >/dev/null
if [[ -e deployment/image-lock.json ]]; then
  echo 'Image is already locked. Reuse it or explicitly create a separate deployment directory.' >&2
  exit 1
fi
docker pull vllm/vllm-openai:v0.22.1
BASE_DIGEST=$(docker image inspect vllm/vllm-openai:v0.22.1 --format '{{index .RepoDigests 0}}')
test -n "$BASE_DIGEST"
docker build --build-arg "VLLM_BASE=$BASE_DIGEST" -f deployment/Dockerfile -t kv-pressure-lab:reproduction .
docker run --rm --entrypoint python3 kv-pressure-lab:reproduction -c 'import importlib.metadata; assert importlib.metadata.version("vllm") == "0.22.1"'
docker run --rm --gpus all kv-pressure-lab:reproduction --help=all > deployment/serve-help.txt
docker run --rm --entrypoint python3 kv-pressure-lab:reproduction -c 'import importlib.metadata,json; print(json.dumps({d.metadata["Name"]:d.version for d in importlib.metadata.distributions()},sort_keys=True,indent=2))' > deployment/python-packages.json
python3 - <<'PY'
from pathlib import Path
text=Path('deployment/serve-help.txt').read_text()
flags=['--kv-cache-memory-bytes','--stream-interval','--revision','--tokenizer-revision','--no-async-scheduling']
missing=[flag for flag in flags if flag not in text]
if missing: raise SystemExit('Pinned CLI missing flags: '+str(missing))
PY
BASE_DIGEST="$BASE_DIGEST" python3 - <<'PY'
import hashlib,json,os,subprocess
from pathlib import Path
image=json.loads(subprocess.check_output(['docker','image','inspect','kv-pressure-lab:reproduction'],text=True))[0]
source=json.loads(Path('vendor/provenance.json').read_text())
record={'base_digest':os.environ['BASE_DIGEST'],'image_id':image['Id'],'vllm_version':'0.22.1',
        'upstream':source,'dockerfile_sha256':hashlib.sha256(Path('deployment/Dockerfile').read_bytes()).hexdigest(),
        'python_packages':json.loads(Path('deployment/python-packages.json').read_text())}
Path('deployment/image-lock.json').write_text(json.dumps(record,indent=2))
PY
