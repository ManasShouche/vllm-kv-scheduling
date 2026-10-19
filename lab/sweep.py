"""Run the frozen 18-arm pilot on a single EC2 GPU, one server at a time."""
import argparse
import json
import random
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from .aws_guard import identity
from .activation import assess


def healthy():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def select_arms(config, one_arm=False, policy=None, capacity=None, seed=None):
    """Select only arms present in the frozen workload; never regenerate it."""
    if one_arm and any(value is not None for value in (policy, capacity, seed)):
        raise ValueError("--one-arm cannot be combined with explicit arm filters")
    for value, allowed, label in ((policy, config["policies"], "policy"),
                                  (capacity, config["kv_capacities_bytes"], "capacity"),
                                  (seed, config["seeds"], "seed")):
        if value is not None and value not in allowed:
            raise ValueError(f"Unknown {label} {value!r} in frozen workload")
    arms = [{"policy": p, "capacity": c, "seed": s}
            for s in config["seeds"] for c in config["kv_capacities_bytes"]
            for p in config["policies"]
            if (policy is None or p == policy) and (capacity is None or c == capacity)
            and (seed is None or s == seed)]
    random.Random(20261005).shuffle(arms)
    if one_arm:
        arms = [arm for arm in arms if arm == {
            "policy": "fixed_2k", "capacity": "roomy", "seed": config["seeds"][0]}]
    if not arms:
        raise ValueError("No matching arms in frozen workload")
    return arms


def main():
    def terminate(signum, frame):
        raise SystemExit("Pilot terminated; preserving logs and stopping its container")
    signal.signal(signal.SIGTERM, terminate)
    p = argparse.ArgumentParser()
    p.add_argument("--workload", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--lock", default="deployment/image-lock.json")
    p.add_argument("--startup-timeout-s", type=float, default=600)
    p.add_argument("--max-hours", type=float, default=4)
    p.add_argument("--one-arm", action="store_true", help="First-arm validation before full rental experiment")
    p.add_argument("--policy", help="Filter policies in the frozen workload")
    p.add_argument("--capacity", help="Filter KV capacities in the frozen workload")
    p.add_argument("--seed", type=int, help="Filter seeds in the frozen workload")
    p.add_argument("--pilot", action="store_true", help="Require pilot readiness audit for selected arms")
    p.add_argument("--dry-run", action="store_true", help="Print selected arms without AWS or Docker access")
    p.add_argument('--trace-overhead', action='store_true', help='One workload arm in off/on/on/off order')
    p.add_argument('--trace-all', action='store_true', help='Trace every matched arm with synchronous scheduling')
    p.add_argument('--comparison-screen', action='store_true', help='Keep valid timeout outcomes and continue later arms; integrity failures still stop')
    a = p.parse_args()
    if a.comparison_screen and (a.pilot or a.one_arm):
        p.error('Comparison screen cannot use pilot readiness gating')
    if a.trace_all and a.trace_overhead:
        p.error('Choose trace-all or trace-overhead')
    if a.max_hours <= 0 or a.startup_timeout_s <= 0:
        raise SystemExit("Timeouts must be positive")
    config = json.loads((Path(a.workload)/"config.json").read_text())
    try:
        arms = select_arms(config, a.one_arm, a.policy, a.capacity, a.seed)
    except ValueError as exc:
        p.error(str(exc))
    if a.trace_overhead:
        if len(arms) != 1:
            p.error('Telemetry overhead requires exactly one selected policy/capacity/seed')
        arms = [{**arms[0], 'trace_enabled': enabled, 'replicate': i}
                for i, enabled in enumerate((False, True, True, False))]
    if a.trace_all:
        arms = [{**arm, 'trace_enabled': True} for arm in arms]
    sync_all = a.trace_overhead or a.trace_all or any(
        arm['policy'] in ('reserve_static', 'reserve_feedback', 'cap_2', 'evict_cost', 'evict_fair') for arm in arms)
    if a.dry_run:
        print(json.dumps({"arms": arms, "pilot_audit": a.pilot or a.one_arm}, indent=2))
        return
    host = identity()
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=False)
    (root/"order.json").write_text(json.dumps({"host": host, "arms": arms}, indent=2))
    deadline = time.monotonic() + a.max_hours*3600
    for index, arm in enumerate(arms):
        name = "kv-pressure-lab-server"
        arm_dir = root/f"{index:02d}-{arm['policy']}-{arm['capacity']}-s{arm['seed']}"
        arm_dir.mkdir()
        status = {"arm": arm, "state": "starting"}
        started = False
        telemetry = None
        try:
            if time.monotonic() >= deadline:
                raise RuntimeError("Wall-clock budget expired before arm start")
            if healthy():
                raise RuntimeError("Port 8000 is already occupied; will not replace an unrelated service")
            server_args = [sys.executable, "-m", "lab.server", "--lock", a.lock, "--workload", a.workload,
                            "--policy", arm["policy"], "--capacity", arm["capacity"],
                            "--provenance", str(arm_dir/"server.json"), "--name", name]
            if sync_all:
                server_args.append('--sync-scheduling')
            if arm.get('trace_enabled'):
                server_args += ['--trace-dir', str(arm_dir/'trace')]
            subprocess.run(server_args, check=True)
            started = True
            with (arm_dir/"gpu.csv").open("w") as gpu_file:
                telemetry = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,uuid,name,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw,pstate,utilization.gpu,memory.used,clocks_throttle_reasons.active",
                                              "--format=csv", "-l", "1"], stdout=gpu_file, stderr=subprocess.STDOUT)
                startup_end = min(deadline, time.monotonic()+a.startup_timeout_s)
                while not healthy():
                    if telemetry.poll() is not None:
                        raise RuntimeError("GPU telemetry exited; inspect gpu.csv and driver query support")
                    if time.monotonic() >= startup_end:
                        raise RuntimeError("Server readiness deadline exceeded")
                    inspect = subprocess.check_output(["docker", "inspect", "--format", "{{.State.Running}}", name], text=True).strip()
                    if inspect != "true":
                        raise RuntimeError("GPU container exited during startup; inspect server.log")
                    time.sleep(2)
                # wall-clock timeout includes generation/drain, and collection handles cancellation.
                subprocess.run([sys.executable, "-m", "lab.collect", "--workload", a.workload,
                                "--seed", str(arm["seed"]), "--output", str(arm_dir/"data"),
                                "--policy", arm["policy"], "--capacity", arm["capacity"],
                                "--server-provenance", str(arm_dir/"server.json")],
                               timeout=max(1, deadline-time.monotonic()), check=True)
                if telemetry.poll() is not None:
                    raise RuntimeError("GPU telemetry exited during measurement; inspect gpu.csv")
                status["state"] = "collected"
        except BaseException as exc:
            status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if telemetry:
                telemetry.terminate()
                telemetry.wait(timeout=10)
            if started:
                with (arm_dir/"server.log").open("w") as log:
                    subprocess.run(["docker", "logs", name], stdout=log, stderr=subprocess.STDOUT)
                subprocess.run(["docker", "stop", "--time", "10", name], check=False)
                # Remove only this experiment's disposable container, after preserving logs.
                subprocess.run(["docker", "rm", name], check=False)
            (arm_dir/"status.json").write_text(json.dumps(status, indent=2))
        # Stop inference and persist server logs before CPU reporting. Every
        # postprocessing command is bounded by the remaining sweep deadline.
        try:
            activation = assess((arm_dir/"server.log").read_text(), arm["policy"])
            (arm_dir/"activation.json").write_text(json.dumps(activation, indent=2))
            if not activation["activation_verified"]:
                raise RuntimeError("Scheduler activation evidence missing or contradictory")
            commands = [[sys.executable, "-m", "lab.analyze", str(arm_dir/"data")],
                        [sys.executable, "-m", "lab.audit", str(arm_dir/"data"), "--workload", a.workload]]
            if a.one_arm or a.pilot:
                commands[1].append("--pilot")
            commands.append([sys.executable, "-m", "lab.analyze", str(arm_dir/"data"), "--plots"])
            if arm.get('trace_enabled'):
                commands.append([sys.executable, '-m', 'lab.trace_audit', str(arm_dir), '--workload', a.workload]
                                + (['--allow-censored'] if a.comparison_screen else []))
            for command in commands:
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("Wall-clock budget expired before reporting")
                subprocess.run(command, timeout=min(120, remaining), check=True)
            status["state"] = "complete"
        except BaseException as exc:
            status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            (arm_dir/"status.json").write_text(json.dumps(status, indent=2))
    print("Pilot finished. EC2 is still running: stop it after copying results.")


if __name__ == "__main__":
    main()
