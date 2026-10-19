"""Detach the dedicated EC2 pilot and arm a host shutdown timer first."""
import argparse
import json
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .aws_guard import identity


def launch(minutes=120):
    if isinstance(minutes, bool) or not isinstance(minutes, int) or not 1 <= minutes <= 120:
        raise ValueError("First-arm host lifetime must be 1–120 minutes")
    host = identity()  # Refuse office Macs and non-EC2 Linux hosts before mutation.
    root = Path(__file__).resolve().parent.parent
    for name in ("aws", "sudo", "systemd-run", "systemctl", "shutdown", "bash"):
        if shutil.which(name) is None:
            raise RuntimeError(f"Required host command missing: {name}")
    subprocess.run(["sudo", "-n", "true"], check=True)
    # Verify stop behavior through AWS before any host shutdown is armed. A host
    # configured to terminate could lose its root disk and experiment artifacts.
    try:
        behavior = subprocess.check_output([
            "aws", "ec2", "describe-instance-attribute", "--region", host["region"],
            "--instance-id", host["instance_id"], "--attribute", "instanceInitiatedShutdownBehavior",
            "--query", "InstanceInitiatedShutdownBehavior.Value", "--output", "text", "--no-cli-pager"
        ], text=True, stderr=subprocess.PIPE).strip()
    except subprocess.CalledProcessError:
        raise RuntimeError("AWS shutdown behavior could not be verified with the existing host credentials") from None
    if behavior != "stop":
        raise RuntimeError("Dedicated pilot host must have instance-initiated shutdown behavior set to stop")
    active = subprocess.check_output([
        "systemctl", "list-units", "--state=activating,active,deactivating", "--no-legend", "--plain",
        "kv-pressure-pilot-*.service"
    ], text=True).strip()
    if active:
        raise RuntimeError("A supervised pilot is already active; will not launch a duplicate")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:8]
    unit = "kv-pressure-pilot-"+run_id
    deadline = "kv-pressure-deadline-"+run_id
    output = root / "handoff-results"
    output.mkdir(exist_ok=True)
    record_path = output / f"{run_id}-supervisor.json"
    log = output / f"{run_id}-supervisor.log"
    record = {"run_id": run_id, "host": host, "unit": unit, "deadline_unit": deadline,
              "maximum_host_runtime_minutes": minutes, "approved_project_budget_usd": 30,
              "shutdown_behavior": behavior, "state": "preparing",
              "budget_scope": "Limits pilot host runtime; storage, network and taxes are not a hard dollar cap",
              "log": str(log)}
    record_path.write_text(json.dumps(record, indent=2))
    shutdown = shutil.which("shutdown")
    bash = shutil.which("bash")
    try:
        # Independent system timer survives client/SSH/SSM disconnects and job failure.
        subprocess.run(["sudo", "-n", "systemd-run", "--unit", deadline,
                        f"--on-active={minutes}m", "--timer-property=AccuracySec=1s",
                        "--", shutdown, "-h", "now"], check=True)
        subprocess.run(["sudo", "-n", "systemctl", "is-active", "--quiet", deadline+".timer"], check=True)
        subprocess.run(["sudo", "-n", "systemd-run", "--unit", unit,
                        "--property=Type=exec", f"--property=WorkingDirectory={root}",
                        f"--property=StandardOutput=append:{log}", f"--property=StandardError=append:{log}",
                        f"--setenv=RUN_ID={run_id}", "--", bash,
                        str(root / "scripts/run_first_arm_then_stop.sh")], check=True)
        record["state"] = "started"
    except subprocess.CalledProcessError:
        record["state"] = "launch_failed_shutdown_requested"
        # Keep any armed deadline in place and also request an immediate stop.
        subprocess.run(["sudo", "-n", shutdown, "-h", "now"], check=False)
        raise
    finally:
        record_path.write_text(json.dumps(record, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=int, default=120)
    args = parser.parse_args()
    print(json.dumps(launch(args.minutes), indent=2))


if __name__ == "__main__":
    main()
