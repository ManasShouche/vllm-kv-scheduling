"""Validate raw measurement artifacts before comparing or extending an AWS pilot."""
import argparse
import json
import math
from pathlib import Path

from .analyze import extract_metrics, summarize
from .workload import digest, schedule_records_digest, validate


STATUSES = {"ok", "http_error", "truncated", "empty", "length_mismatch", "censored",
            "timeout", "client_error", "client_capacity_rejected"}


def load_artifacts(path):
    path = Path(path)
    meta = json.loads((path / "run.json").read_text())
    rows = [json.loads(line) for line in (path / "requests.jsonl").read_text().splitlines()]
    metrics = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
    return meta, rows, metrics


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Nonfinite or nonnumeric {name}")
    return value


def verify_records(meta, rows, metrics, workload=None):
    """Reject missing/altered requests and impossible successful streams."""
    config = meta["config"]
    duration = validate(config)
    elapsed = finite(meta["observed_duration_s"], "observed_duration_s")
    if elapsed < duration:
        raise ValueError("Observation ends before the offered workload")
    if not meta.get("complete") or len(rows) != meta.get("scheduled_requests") or len(rows) != meta.get("recorded_requests"):
        raise ValueError("Incomplete request accounting")
    if not rows or len(rows) != len({r["request_id"] for r in rows}):
        raise ValueError("Empty or duplicate request records")
    if schedule_records_digest(rows) != meta.get("schedule_records_sha256"):
        raise ValueError("Request metadata differs from scheduled request fingerprint")
    boundaries = []
    offset = 0
    for phase in config["phases"]:
        boundaries.append((offset, offset + phase["duration_s"], phase))
        offset += phase["duration_s"]
    for row in rows:
        arrival = finite(row["arrival_s"], "arrival_s")
        phase = next((p for lo, hi, p in boundaries if lo <= arrival < hi), None)
        if phase is None or row["phase"] != phase["name"]:
            raise ValueError("Request phase differs from arrival window")
        if row["warmup"] != (arrival < config["warmup_s"]):
            raise ValueError("Request warmup label differs from arrival time")
        if any(row[key] not in phase.get(key+"_choices", [phase[key]]) for key in ("prompt_tokens", "output_tokens")):
            raise ValueError("Request lengths differ from configured phase")
        if row["status"] not in STATUSES:
            raise ValueError("Unknown request status")
        finish = finite(row["finished_s"], "finished_s")
        if not arrival <= finish <= elapsed + 1e-6:
            raise ValueError("Request completion outside observation window")
        ticks = row["event_times_s"]
        sent = row.get("actual_dispatch_s")
        if row["status"] == "client_capacity_rejected":
            if sent is not None or row.get("dispatch_lag_s") is not None or ticks:
                raise ValueError("Rejected request has dispatch or content events")
            continue
        sent = finite(sent, "actual_dispatch_s")
        lag = finite(row["dispatch_lag_s"], "dispatch_lag_s")
        if sent < arrival - 1e-6 or finish < sent or not math.isclose(lag, sent-arrival, abs_tol=1e-6):
            raise ValueError("Invalid dispatch/completion timing")
        last = sent
        for tick in ticks:
            tick = finite(tick, "event time")
            if not last <= tick <= finish:
                raise ValueError("Content events are out of order or outside request lifetime")
            last = tick
        if row["status"] == "ok":
            if (row.get("http_status") != 200 or not row.get("stream_done") or
                    not row.get("finish_reason") or not ticks or
                    row.get("completion_tokens") != row["output_tokens"]):
                raise ValueError("Successful request lacks valid finish, usage or content")
    last = -1
    for sample in metrics:
        tick = finite(sample["t_s"], "metrics time")
        if not last <= tick <= elapsed or tick < 0:
            raise ValueError("Metrics timestamps outside ordered observation window")
        last = tick
        values = extract_metrics(sample.get("raw_prometheus", ""))
        for key, value in values.items():
            finite(value, key)
            if value < 0 or (key == "kv_usage" and value > 1):
                raise ValueError("Server metric outside expected nonnegative/fraction range")
    if workload is not None:
        root = Path(workload)
        manifest = json.loads((root / "manifest.json").read_text())
        frozen_config = json.loads((root / "config.json").read_text())
        seed = meta["labels"]["seed"]
        schedule_path = root / f"seed-{seed}.jsonl"
        if frozen_config != config or digest(root / "config.json") != manifest["config_sha256"]:
            raise ValueError("Frozen workload configuration mismatch")
        if digest(schedule_path) != manifest["schedule_sha256"][schedule_path.name]:
            raise ValueError("Frozen schedule hash mismatch")
        scheduled = [json.loads(line) for line in schedule_path.read_text().splitlines()]
        if schedule_records_digest(scheduled) != schedule_records_digest(rows):
            raise ValueError("Raw request records differ from frozen workload")
        provenance = meta["provenance"]
        server = provenance["server"]
        labels = meta["labels"]
        if provenance["workload"] != manifest or provenance["schedule_sha256"] != digest(schedule_path):
            raise ValueError("Workload provenance mismatch")
        if server["model"] != config["model"] or server["model_revision"] != manifest["model_revision"]:
            raise ValueError("Server model pin mismatch")
        if server["policy"] != labels["policy"] or server["capacity"] != labels["capacity"]:
            raise ValueError("Server arm labels mismatch")
        if server["kv_capacity_bytes"] != config["kv_capacities_bytes"][labels["capacity"]]:
            raise ValueError("Server KV capacity mismatch")


def audit(path, workload=None, pilot=False):
    try:
        meta, rows, metrics = load_artifacts(path)
        verify_records(meta, rows, metrics, workload)
        summary = summarize(meta, rows, metrics)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        return {"artifact_valid": False, "ready": False, "blockers": [str(exc)]}
    blockers = list(summary["quality_flags"])
    if pilot and any(row["status"] != "ok" for row in rows):
        blockers.append("pilot_has_unsuccessful_requests")
    if pilot and any(len(row["event_times_s"]) < 2 for row in rows):
        blockers.append("pilot_lacks_observable_streaming_gaps")
    if pilot:
        for name, phase in summary["phases"].items():
            for flag in phase["quality_flags"]:
                if flag in {"no_phase_kv_samples", "insufficient_phase_preemption_samples"}:
                    blockers.append(name+":"+flag)
    # Sparse tails and growing queues describe the workload; neither is corruption.
    return {"artifact_valid": True, "ready": not blockers, "blockers": blockers,
            "scheduled_requests": len(rows), "pilot_mode": pilot,
            "unsuccessful_requests": sum(row["status"] != "ok" for row in rows),
            "all_requests_succeeded": all(row["status"] == "ok" for row in rows),
            "workload_verified": workload is not None,
            "advisories": {name: phase["quality_flags"] for name, phase in summary["phases"].items()},
            "scope": "Collection integrity and pilot readiness; not scheduler correctness or a causal result"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory")
    parser.add_argument("--workload")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    report = audit(args.run_directory, args.workload, args.pilot)
    (Path(args.run_directory) / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ready"] else 1)


if __name__ == "__main__":
    main()
