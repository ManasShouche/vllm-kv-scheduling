"""Apply the unmodified attributed patch only to exactly vLLM 0.22.1."""
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
from pathlib import Path

if importlib.metadata.version("vllm") != "0.22.1":
    raise SystemExit("Refusing to patch any other vLLM version")
patch = Path("/opt/reproduction/ppas.patch")
source = json.loads(Path("/opt/reproduction/provenance.json").read_text())
if hashlib.sha256(patch.read_bytes()).hexdigest() != source["patch_sha256"]:
    raise SystemExit("Vendor patch hash mismatch")
spec = importlib.util.find_spec("vllm")
root = Path(spec.origin).parent.parent
subprocess.run(["git", "apply", "--check", str(patch)], cwd=root, check=True)
subprocess.run(["git", "apply", str(patch)], cwd=root, check=True)
scheduler = root / "vllm/v1/core/sched/scheduler.py"
ast.parse(scheduler.read_text())
if 'os.getenv("PPAS_ENABLED", "0")' not in scheduler.read_text():
    raise SystemExit("P-PAS activation marker missing")
