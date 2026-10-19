"""Launch one private EC2 container arm with an immutable image and fixed model."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from .aws_guard import identity
from .workload import validate, POLICY_BUDGETS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lock", default="deployment/image-lock.json")
    p.add_argument("--workload", required=True)
    p.add_argument("--policy", choices=list(POLICY_BUDGETS), required=True)
    p.add_argument("--capacity", choices=["roomy", "constrained"], required=True)
    p.add_argument("--provenance", required=True)
    p.add_argument("--name", default="kv-pressure-server")
    p.add_argument('--trace-dir', help='Dedicated host directory for scheduler telemetry')
    p.add_argument('--sync-scheduling', action='store_true')
    a = p.parse_args()
    if a.policy in ('reserve_static', 'reserve_feedback'):
        a.sync_scheduling = True
    if a.policy in ('cap_2', 'evict_cost', 'evict_fair'):
        a.sync_scheduling = True
    host = identity()
    root = Path(a.workload)
    config = json.loads((root / "config.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    validate(config)
    lock = json.loads(Path(a.lock).read_text())
    inspect = json.loads(subprocess.check_output(["docker", "image", "inspect", lock["image_id"]], text=True))[0]
    if inspect["Id"] != lock["image_id"]:
        raise RuntimeError("Docker image differs from lock")
    command = ["docker", "run", "-d", "--gpus", "all", "--ipc", "host", "--network", "host",
               "--name", a.name, "-e", f"PPAS_ENABLED={int(a.policy in ('ppas', 'decode_guard'))}",
               "-e", f"KVLAB_DECODE_GUARD={int(a.policy == 'decode_guard')}",
               "-v", f"{Path('deployment/model-cache').resolve()}:/root/.cache/huggingface"]
    if a.policy in ('reserve_static', 'reserve_feedback'):
        command += ['-e', f'KVLAB_HEADROOM={a.policy}']
    if a.policy in ('evict_cost', 'evict_fair'):
        command += ['-e', f'KVLAB_EVICTION={a.policy}']
    if os.getenv("HF_TOKEN"):
        command += ["-e", "HF_TOKEN"]
    if a.trace_dir:
        if not a.sync_scheduling:
            raise ValueError('Recomputation telemetry currently requires synchronous scheduling')
        trace = Path(a.trace_dir).resolve()
        trace.mkdir(parents=True, exist_ok=False)
        command += ['-v', f'{trace}:/lab-trace', '-e', 'KVLAB_TRACE_DIR=/lab-trace']
    command += [lock["image_id"], config["model"], "--host", "127.0.0.1", "--port", "8000",
                "--revision", manifest["model_revision"], "--tokenizer-revision", manifest["model_revision"],
                "--dtype", "bfloat16", "--max-model-len", str(config["max_model_len"]),
                "--max-num-batched-tokens", str(config["policies"][a.policy]),
                "--max-num-seqs", str(2 if a.policy == "cap_2" else config["max_num_seqs"]),
                "--kv-cache-memory-bytes", str(config["kv_capacities_bytes"][a.capacity]),
                "--no-enable-prefix-caching", "--enable-chunked-prefill", "--stream-interval", "1",
                "--seed", "0"]
    if a.sync_scheduling:
        command += ['--no-async-scheduling']
    container = subprocess.check_output(command, text=True).strip()
    data = {**lock, **host, "container_id": container, "model": config["model"],
            "model_revision": manifest["model_revision"], "policy": a.policy, "capacity": a.capacity,
            "command": command, "async_scheduling": "pinned v0.22.1 default; verify startup log",
            "kv_capacity_bytes": config["kv_capacities_bytes"][a.capacity],
            "trace_enabled": bool(a.trace_dir)}
    if a.sync_scheduling:
        data['async_scheduling'] = 'explicitly disabled for telemetry and matched controls'
    path = Path(a.provenance)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    print(container)


if __name__ == "__main__":
    main()
