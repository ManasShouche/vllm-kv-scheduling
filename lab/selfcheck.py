"""CPU fixture: exercise collection, auditing, plotting and comparison.

No model, GPU, AWS instance or scheduler is executed. Fixture responses/metrics
are invented protocol test data and must never be interpreted as measurements.
"""
import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

from aiohttp import web

from .audit import audit
from .collect import run
from .workload import digest


async def exercise(output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path("configs/pilot.json").read_text())
    config.update(warmup_s=0, seeds=[41], request_timeout_s=2, drain_timeout_s=1,
                  dispatch_lag_limit_s=.1)
    for index, phase in enumerate(config["phases"]):
        phase.update(duration_s=1.6, request_rate=2.5, prompt_tokens=3, output_tokens=16*(index+1))
    workload = root / "workload"
    workload.mkdir()
    (workload / "config.json").write_text(json.dumps(config, indent=2))
    schedule = []
    for index, phase in enumerate(config["phases"]):
        for arrival in (.02, .2, .55, .9):
            schedule.append({"request_id": f"fixture-{len(schedule)}", "arrival_s": index*1.6+arrival,
                             "phase": phase["name"], "warmup": False, "prompt_tokens": 3,
                             "output_tokens": phase["output_tokens"], "prompt_token_ids": [1, 2, 3]})
    schedule_path = workload / "seed-41.jsonl"
    schedule_path.write_text("".join(json.dumps(row)+"\n" for row in schedule))
    manifest = {"model": config["model"], "model_revision": "0"*40,
                "config_sha256": digest(workload / "config.json"),
                "schedule_sha256": {schedule_path.name: digest(schedule_path)},
                "workload_kind": "cpu_fixture_not_inference"}
    (workload / "manifest.json").write_text(json.dumps(manifest, indent=2))
    state = {"origin": 0}

    async def health(request):
        state["origin"] = time.perf_counter()
        return web.Response(text="CPU fixture only")

    async def metrics(request):
        late = time.perf_counter()-state["origin"] >= 1.6
        return web.Response(text=("# CPU fixture metrics; not GPU data\n"
                                  "vllm:num_requests_running 1\nvllm:num_requests_waiting 0\n"
                                  f"vllm:kv_cache_usage_perc {0.95 if late else 0.2}\n"
                                  f"vllm:num_preemptions_total {2 if late else 0}\n"))

    async def completion(request):
        body = await request.json()
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for text in ("fixture ", "protocol ", "response"):
            message = ("data: "+json.dumps({"choices": [{"text": text}]})+"\r\n\r\n").encode()
            for fragment in (message[:7], message[7:]):
                await response.write(fragment)
            await asyncio.sleep(.005)
        message = {"choices": [{"text": "", "finish_reason": "length"}],
                   "usage": {"completion_tokens": body["max_tokens"]}}
        await response.write(("data: "+json.dumps(message)+"\n\ndata: [DONE]\n\n").encode())
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_get("/metrics", metrics)
    app.router.add_post("/v1/completions", completion)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    matrix = root / "matrix"
    matrix.mkdir()
    reports = []
    try:
        for policy in config["policies"]:
            for capacity in config["kv_capacities_bytes"]:
                target = matrix / f"{policy}-{capacity}" / "data"
                labels = {"policy": policy, "capacity": capacity, "seed": 41}
                server = {"model": config["model"], "model_revision": manifest["model_revision"],
                          "policy": policy, "capacity": capacity, "image_id": "cpu-fixture-no-image",
                          "instance_id": "cpu-fixture-no-ec2", "kv_capacity_bytes": config["kv_capacities_bytes"][capacity]}
                provenance = {"workload": manifest, "schedule_sha256": digest(schedule_path),
                              "server": server, "execution_kind": "cpu_fixture"}
                await run(f"http://127.0.0.1:{port}", schedule, config, target, labels, provenance)
                subprocess.run([sys.executable, "-m", "lab.analyze", str(target), "--plots"],
                               check=True, stdout=subprocess.DEVNULL)
                report = audit(target, workload, pilot=True)
                (target / "audit.json").write_text(json.dumps(report, indent=2))
                if not report["ready"]:
                    raise RuntimeError(f"CPU fixture audit failed: {report}")
                reports.append(report)
    finally:
        await runner.cleanup()
    subprocess.run([sys.executable, "-m", "lab.compare", str(matrix), "--workload", str(workload)], check=True)
    (root / "selfcheck.json").write_text(json.dumps({"execution_kind": "cpu_fixture", "arms": len(reports),
        "requests": sum(r["scheduled_requests"] for r in reports), "all_audits_ready": True,
        "scope": "Protocol, artifact and report integration only; no inference or AWS execution"}, indent=2))
    print(root / "selfcheck.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    asyncio.run(exercise(args.output))


if __name__ == "__main__":
    main()
