"""Descriptive pilot statistics and plots. No IID token bootstrap or SOTA claims."""
import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np

from .workload import validate


METRICS = {
    "running": ["vllm:num_requests_running"],
    "waiting": ["vllm:num_requests_waiting"],
    "kv_usage": ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
    "preemptions": ["vllm:num_preemptions_total"],
}


def extract_metrics(text):
    raw = collections.defaultdict(list)
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        match = re.match(r'^([^\s{]+)(?:\{[^}]*\})?\s+([-+0-9.eE]+)(?:\s+\d+)?$', line)
        if match:
            raw[match[1]].append(float(match[2]))
    found = {}
    for key, aliases in METRICS.items():
        for name in aliases:
            if name in raw:
                values = raw[name]
                found[key] = sum(values) if key != "kv_usage" else max(values)
                break
    return found


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def request_stats(row):
    ticks = row.get("event_times_s", [])
    gaps = [b-a for a, b in zip(ticks, ticks[1:])]
    return {"ttft_dispatch_s": ticks[0]-row["actual_dispatch_s"] if ticks else None,
            "ttft_arrival_s": ticks[0]-row["arrival_s"] if ticks else None,
            "max_event_gap_s": max(gaps) if gaps else None,
            "event_gaps_s": gaps}


def summarize(meta, rows, metrics):
    config = meta["config"]
    duration = validate(config)
    if len(rows) != len({r["request_id"] for r in rows}):
        raise ValueError("Duplicate request records")
    output = {"labels": meta["labels"], "complete": meta.get("complete", False),
              "record_count": len(rows), "phases": {}, "quality_flags": [],
              "uncertainty": "Per-seed descriptive statistics only; no confidence intervals",
              "metric_semantics": "SSE nonempty text event gaps, not token ITL"}
    if not meta.get("complete") or len(rows) != meta.get("scheduled_requests"):
        output["quality_flags"].append("incomplete_request_accounting")
    lags = [r["dispatch_lag_s"] for r in rows if r.get("dispatch_lag_s") is not None]
    output["dispatch_lag_p99_s"] = percentile(lags, 99)
    if lags and output["dispatch_lag_p99_s"] > config["dispatch_lag_limit_s"]:
        output["quality_flags"].append("open_loop_dispatch_lag")
    if any(r["status"] == "client_capacity_rejected" for r in rows):
        output["quality_flags"].append("client_capacity_limit")
    matched = [extract_metrics(m.get("raw_prometheus", "")) for m in metrics]
    output["metrics_discovered"] = sorted(set().union(*(set(m) for m in matched))) if matched else []
    missing = set(METRICS) - set(output["metrics_discovered"])
    if missing:
        output["quality_flags"].append("missing_server_metrics:" + ",".join(sorted(missing)))
    if any("error" in m for m in metrics):
        output["quality_flags"].append("metrics_scrape_errors")
    preemptions = [m["preemptions"] for m in matched if "preemptions" in m]
    output["preemption_delta"] = preemptions[-1]-preemptions[0] if preemptions else None
    if any(b < a for a, b in zip(preemptions, preemptions[1:])):
        output["quality_flags"].append("preemption_counter_reset")
    offset = 0.0
    for phase in config["phases"]:
        end = offset + phase["duration_s"]
        cohort = [r for r in rows if r["phase"] == phase["name"] and not r["warmup"]]
        success = [r for r in cohort if r["status"] == "ok"]
        stats = [request_stats(r) for r in success]
        ttfts = [s["ttft_dispatch_s"] for s in stats]
        stutters = [s["max_event_gap_s"] for s in stats if s["max_event_gap_s"] is not None]
        passed = sum(s["ttft_dispatch_s"] <= config["ttft_slo_s"] and
                     s["max_event_gap_s"] is not None and
                     s["max_event_gap_s"] <= config["event_gap_slo_s"] for s in stats)
        measured_s = phase["duration_s"] - (config["warmup_s"] if offset == 0 else 0)
        by_window = [r for r in rows if r["status"] == "ok" and
                     max(offset, config["warmup_s"]) <= r["finished_s"] < end]
        item = {"arrivals": len(cohort), "completed": len(success),
                "status_counts": dict(collections.Counter(r["status"] for r in cohort)),
                "slo_attainment_all_arrivals": passed / len(cohort) if cohort else None,
                "slo_passing_requests": passed,
                "cohort_slo_passes_per_offered_second": passed / measured_s,
                "window_completed_requests_per_s": len(by_window) / measured_s,
                "cohort_completion_tokens": sum(r["completion_tokens"] for r in success),
                "ttft_p50_s": percentile(ttfts, 50), "ttft_p95_s": percentile(ttfts, 95),
                "ttft_p99_s": percentile(ttfts, 99),
                "request_max_event_gap_p50_s": percentile(stutters, 50),
                "request_max_event_gap_p99_s": percentile(stutters, 99),
                "completed_after_arrival_phase": sum(r["finished_s"] >= end for r in success),
                "p99_expected_tail_samples": len(success) / 100,
                "quality_flags": []}
        if len(success) < 1000:
            item["quality_flags"].append("sparse_p99_exploratory_only")
        waits = [(m["t_s"], parsed["waiting"]) for m, parsed in zip(metrics, matched)
                 if offset <= m["t_s"] < end and "waiting" in parsed]
        phase_metrics = [parsed for m, parsed in zip(metrics, matched) if offset <= m["t_s"] < end]
        kv = [m["kv_usage"] for m in phase_metrics if "kv_usage" in m]
        counter = [m["preemptions"] for m in phase_metrics if "preemptions" in m]
        item["kv_usage_peak_fraction"] = max(kv) if kv else None
        item["kv_usage_sample_count"] = len(kv)
        item["kv_samples_at_or_above_90pct_fraction"] = sum(v >= .9 for v in kv)/len(kv) if kv else None
        item["preemption_delta_between_in_phase_samples"] = counter[-1]-counter[0] if len(counter) >= 2 else None
        item["pressure_semantics"] = "Scrape-sampled KV fraction; 90% is a descriptive threshold. Counter delta omits unsampled boundaries."
        if not kv:
            item["quality_flags"].append("no_phase_kv_samples")
        if len(counter) < 2:
            item["quality_flags"].append("insufficient_phase_preemption_samples")
        if len(waits) >= 10:
            x, y = zip(*waits)
            item["waiting_queue_slope_per_s"] = float(np.polyfit(x, y, 1)[0])
            quarter = max(1, len(y)//4)
            if np.mean(y[-quarter:]) > np.mean(y[:quarter]) + 2 and item["waiting_queue_slope_per_s"] > 0.01:
                item["quality_flags"].append("growing_queue_nonstationary")
        output["phases"][phase["name"]] = item
        offset = end
    # Drain-inclusive output throughput: no tail requests hidden from denominator.
    useful = [r for r in rows if not r["warmup"] and r["status"] == "ok"]
    elapsed = max(meta["observed_duration_s"]-config["warmup_s"], 1e-9)
    output["completed_output_tokens_per_drain_inclusive_s"] = sum(r["completion_tokens"] for r in useful) / elapsed
    return output


def plots(path, meta, rows, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    arrivals, ttfts, colors, gap_times, gaps = [], [], [], [], []
    for row in rows:
        stat = request_stats(row)
        if stat["ttft_dispatch_s"] is not None:
            arrivals.append(row["arrival_s"])
            ttfts.append(stat["ttft_dispatch_s"])
            colors.append("tab:blue" if row["status"] == "ok" else "tab:red")
        gap_times.extend(row.get("event_times_s", [])[1:])
        gaps.extend(stat["event_gaps_s"])
    # One collection per panel: a real pilot can contain hundreds of thousands
    # of events. One scatter artist per event made reporting consume GPU-hours.
    axes[0].scatter(arrivals, ttfts, s=9, c=colors)
    axes[1].scatter(gap_times, gaps, s=3, alpha=.35, c="tab:orange")
    axes[0].set_ylabel("TTFT from dispatch (s)")
    axes[1].set_ylabel("SSE content-event gap (s)")
    axes[0].axhline(meta["config"]["ttft_slo_s"], linestyle="--", c="grey")
    axes[1].axhline(meta["config"]["event_gap_slo_s"], linestyle="--", c="grey")
    for key in ("running", "waiting", "preemptions"):
        points = [(m["t_s"], extract_metrics(m.get("raw_prometheus", "")).get(key)) for m in metrics]
        points = [(x, y) for x, y in points if y is not None]
        if points:
            x, y = zip(*points)
            if key == "preemptions":
                y = [v-y[0] for v in y]
            axes[2].plot(x, y, marker=".", linestyle="none", label=key)
    if axes[2].lines:
        axes[2].legend()
    axes[2].set_ylabel("Requests / preemptions")
    kv = [(m["t_s"], extract_metrics(m.get("raw_prometheus", "")).get("kv_usage")) for m in metrics]
    kv = [(x, y) for x, y in kv if y is not None]
    if kv:
        x, y = zip(*kv)
        axes[3].plot(x, y, marker="o", linestyle="none", color="tab:purple", label="Observed scrapes")
    axes[3].axhline(.9, linestyle="--", c="grey", label="90% descriptive threshold")
    axes[3].set_ylim(0, 1.05)
    axes[3].set_ylabel("Sampled KV usage (fraction)")
    axes[3].legend(fontsize=8)
    axes[3].set_xlabel("Seconds since run start (includes drain)")
    elapsed = 0
    for phase in meta["config"]["phases"][:-1]:
        elapsed += phase["duration_s"]
        for ax in axes:
            ax.axvline(elapsed, c="black", linestyle=":")
    kind = "CPU FIXTURE — no inference" if meta.get("provenance", {}).get("execution_kind") == "cpu_fixture" else "Pilot"
    fig.suptitle(f"{kind}: {meta['labels']} — event gaps are not token ITL")
    fig.tight_layout()
    fig.savefig(path / "timeline.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_directory")
    p.add_argument("--plots", action="store_true")
    a = p.parse_args()
    path = Path(a.run_directory)
    meta = json.loads((path / "run.json").read_text())
    rows = [json.loads(line) for line in (path / "requests.jsonl").read_text().splitlines()]
    metrics = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
    result = summarize(meta, rows, metrics)
    (path / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    if a.plots:
        plots(path, meta, rows, metrics)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
