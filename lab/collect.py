"""Open-loop streaming client. SSE gaps remain event gaps, not GPU token ITL."""
import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

import aiohttp

from .workload import digest, schedule_records_digest, validate, POLICY_BUDGETS


async def sse_data(content):
    """Parse fragmented UTF-8/CRLF SSE, comments, and multiline data fields."""
    lines = []
    while True:
        raw = await content.readline()
        if not raw:
            if lines:
                yield "\n".join(lines)
            return
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if lines:
                yield "\n".join(lines)
                lines = []
        elif line.startswith("data:"):
            lines.append(line[5:].removeprefix(" "))


def content_delta(event):
    choices = event.get("choices") or []
    if not choices:
        return ""
    return choices[0].get("text") or choices[0].get("delta", {}).get("content") or ""


def append_json(handle, value):
    handle.write(json.dumps(value, separators=(",", ":")) + "\n")
    handle.flush()


async def request(session, endpoint, model, row, origin, timeout_s):
    sent = time.perf_counter() - origin
    record = {k: v for k, v in row.items() if k != "prompt_token_ids"}
    record.update(actual_dispatch_s=sent, dispatch_lag_s=sent-row["arrival_s"],
                  status="incomplete", event_times_s=[], output_characters=0,
                  completion_tokens=None, stream_done=False, finish_reason=None)
    payload = {"model": model, "prompt": row["prompt_token_ids"], "max_tokens": row["output_tokens"],
               "request_id": row["request_id"],
               "temperature": 0, "ignore_eos": True, "stream": True,
               "stream_options": {"include_usage": True}}
    try:
        async with asyncio.timeout(timeout_s):
            async with session.post(endpoint + "/v1/completions", json=payload) as resp:
                record["http_status"] = resp.status
                if resp.status != 200:
                    record.update(status="http_error", error=(await resp.text())[:1000])
                    return record
                async for data in sse_data(resp.content):
                    if data.strip() == "[DONE]":
                        record["stream_done"] = True
                        break
                    event = json.loads(data)
                    if event.get('id'):
                        previous = record.setdefault('server_response_id', event['id'])
                        if previous != event['id']:
                            raise ValueError('Response ID changed within stream')
                    if "error" in event:
                        raise RuntimeError(str(event["error"])[:1000])
                    text = content_delta(event)
                    if text:
                        record["event_times_s"].append(time.perf_counter() - origin)
                        record["output_characters"] += len(text)
                    if event.get("usage"):
                        record["completion_tokens"] = event["usage"].get("completion_tokens")
                    choices = event.get("choices") or []
                    if choices and choices[0].get("finish_reason"):
                        record["finish_reason"] = choices[0]["finish_reason"]
                if not record["stream_done"] or not record["finish_reason"]:
                    record.update(status="truncated", error="Stream lacks DONE or finish reason")
                elif not record["event_times_s"]:
                    record.update(status="empty", error="No nonempty text event")
                elif record["completion_tokens"] != row["output_tokens"]:
                    record.update(status="length_mismatch", error="Usage absent or output length differs")
                else:
                    record["status"] = "ok"
    except asyncio.CancelledError:
        record.update(status="censored", error="Global drain deadline")
    except TimeoutError:
        record.update(status="timeout", error="Per-request deadline")
    except Exception as exc:
        record.update(status="client_error", error=f"{type(exc).__name__}: {exc}"[:1000])
    finally:
        record["finished_s"] = time.perf_counter() - origin
    return record


async def metrics(session, endpoint, origin, handle, stop):
    while not stop.is_set():
        tick = time.perf_counter() - origin
        try:
            async with session.get(endpoint + "/metrics", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Metrics HTTP {resp.status}")
                raw = await resp.text()
                append_json(handle, {"t_s": tick, "raw_prometheus": raw})
        except Exception as exc:
            append_json(handle, {"t_s": tick, "error": str(exc)[:500]})
        try:
            await asyncio.wait_for(stop.wait(), timeout=1)
        except TimeoutError:
            pass


async def run(endpoint, schedule, config, dest, labels, provenance):
    duration = validate(config)
    if not schedule:
        raise ValueError("Empty schedule")
    ids = set()
    last = -1
    for row in schedule:
        if row["request_id"] in ids or not 0 <= row["arrival_s"] < duration or row["arrival_s"] < last:
            raise ValueError("Duplicate IDs or invalid arrival ordering")
        if len(row["prompt_token_ids"]) != row["prompt_tokens"]:
            raise ValueError("Prompt length mismatch")
        if row["prompt_tokens"] + row["output_tokens"] >= config["max_model_len"]:
            raise ValueError("Request exceeds reserved context boundary")
        ids.add(row["request_id"])
        last = row["arrival_s"]
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    api_key = os.getenv("VLLM_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    connector = aiohttp.TCPConnector(limit=0)  # no hidden client-side connection queue
    async with aiohttp.ClientSession(connector=connector, headers=headers,
                                    timeout=aiohttp.ClientTimeout(total=None), read_bufsize=1048576) as session:
        async with session.get(endpoint + "/health", timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                raise RuntimeError("Server not healthy")
        origin = time.perf_counter()
        meta = {"schema_version": 1, "labels": labels, "config": config,
                "origin_perf_counter_ns": round(origin*1e9),
                "provenance": provenance, "origin_unix_s": time.time(),
                "metric_semantics": "Client SSE nonempty-text-event gaps; not token ITL",
                "endpoint_kind": "OpenAI completions", "recorded_requests": 0,
                "schedule_records_sha256": schedule_records_digest(schedule),
                "scheduled_requests": len(schedule)}
        (dest / "run.json").write_text(json.dumps(meta, indent=2))
        with (dest / "requests.jsonl").open("w") as records, (dest / "metrics.jsonl").open("w") as mh:
            stop = asyncio.Event()
            scraper = asyncio.create_task(metrics(session, endpoint, origin, mh, stop))
            active = set()
            count = 0

            async def execute(row):
                nonlocal count
                result = await request(session, endpoint, config["model"], row, origin, config["request_timeout_s"])
                append_json(records, result)
                count += 1

            try:
                for row in schedule:
                    await asyncio.sleep(max(0, origin + row["arrival_s"] - time.perf_counter()))
                    active = {t for t in active if not t.done()}
                    if len(active) >= config["max_active_requests"]:
                        # Explicit invalid-run guard; never silently become closed loop.
                        skipped = {k: v for k, v in row.items() if k != "prompt_token_ids"}
                        skipped.update(status="client_capacity_rejected", event_times_s=[],
                                       completion_tokens=None, actual_dispatch_s=None,
                                       dispatch_lag_s=None, finished_s=time.perf_counter()-origin)
                        append_json(records, skipped)
                        count += 1
                    else:
                        active.add(asyncio.create_task(execute(row)))
                await asyncio.sleep(max(0, origin + duration - time.perf_counter()))
                if active:
                    _, pending = await asyncio.wait(active, timeout=config["drain_timeout_s"])
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*active)
            finally:
                for task in active:
                    if not task.done():
                        task.cancel()
                if active:
                    await asyncio.gather(*active, return_exceptions=True)
                stop.set()
                await scraper
                meta.update(recorded_requests=count, scheduled_requests=len(schedule),
                            observed_duration_s=time.perf_counter()-origin,
                            complete=count == len(schedule))
                (dest / "run.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", default="http://127.0.0.1:8000")
    p.add_argument("--workload", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--policy", choices=list(POLICY_BUDGETS), required=True)
    p.add_argument("--capacity", choices=["roomy", "constrained"], required=True)
    p.add_argument("--server-provenance", required=True)
    a = p.parse_args()
    root = Path(a.workload)
    manifest = json.loads((root / "manifest.json").read_text())
    path = root / f"seed-{a.seed}.jsonl"
    if digest(path) != manifest["schedule_sha256"][path.name] or digest(root / "config.json") != manifest["config_sha256"]:
        raise RuntimeError("Workload hashes differ from frozen manifest")
    config = json.loads((root / "config.json").read_text())
    server = json.loads(Path(a.server_provenance).read_text())
    if server.get("model_revision") != manifest["model_revision"] or server.get("model") != config["model"]:
        raise RuntimeError("Server and workload model pins differ")
    if server.get("policy") != a.policy or server.get("capacity") != a.capacity:
        raise RuntimeError("Server labels do not match requested arm")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    provenance = {"workload": manifest, "server": server, "schedule_sha256": digest(path),
                  "execution_kind": "ec2_gpu"}
    print(json.dumps(asyncio.run(run(a.endpoint.rstrip("/"), rows, config, a.output,
                                    {"policy": a.policy, "capacity": a.capacity, "seed": a.seed}, provenance)), indent=2))


if __name__ == "__main__":
    main()
