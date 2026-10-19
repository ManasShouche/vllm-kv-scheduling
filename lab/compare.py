"""Verify matched arms and report per-seed pilot tradeoffs, without invented CIs."""
import argparse
import collections
import json
from pathlib import Path

from .analyze import summarize
from .audit import load_artifacts, verify_records


def load_matrix(root, allow_partial=False, workload=None):
    runs = []
    for path in sorted(Path(root).glob("*/data/summary.json")):
        meta, rows, metrics = load_artifacts(path.parent)
        verify_records(meta, rows, metrics, workload)
        summary = summarize(meta, rows, metrics)
        runs.append((summary, meta))
    if not runs:
        raise ValueError("No analyzed runs")
    first = runs[0][1]
    config = first["config"]
    expected = {(p, c, s) for p in config["policies"] for c in config["kv_capacities_bytes"] for s in config["seeds"]}
    seen = set()
    schedules = {}
    for summary, meta in runs:
        labels = meta["labels"]
        key = (labels["policy"], labels["capacity"], labels["seed"])
        if key in seen or key not in expected:
            raise ValueError("Duplicate or unexpected experiment arm")
        seen.add(key)
        if meta["config"] != config:
            raise ValueError("Configurations differ across arms")
        a, b = meta["provenance"], first["provenance"]
        if a.get("execution_kind") != b.get("execution_kind"):
            raise ValueError("Execution kinds differ across arms")
        if a["workload"] != b["workload"]:
            raise ValueError("Workload manifests differ across arms")
        if a["server"]["image_id"] != b["server"]["image_id"]:
            raise ValueError("Container images differ across arms")
        if a["server"]["instance_id"] != b["server"]["instance_id"]:
            raise ValueError("Physical EC2 instance differs across arms")
        seed = labels["seed"]
        if seed in schedules and schedules[seed] != a["schedule_sha256"]:
            raise ValueError("Arrival schedule differs for same seed")
        schedules[seed] = a["schedule_sha256"]
        if not summary["complete"]:
            raise ValueError("Incomplete request accounting")
    if seen != expected and not allow_partial:
        raise ValueError(f"Incomplete matrix: {len(seen)}/{len(expected)}. Use --allow-partial only for diagnostics.")
    return runs, sorted(expected-seen)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--workload", help="Verify each arm against the frozen prompts, schedule and model pins")
    a = p.parse_args()
    runs, missing = load_matrix(a.root, a.allow_partial, a.workload)
    root = Path(a.root)
    title = "Exploratory pilot comparison" + (" — INCOMPLETE MATRIX" if missing else "")
    if runs[0][1]["provenance"].get("execution_kind") == "cpu_fixture":
        title = "CPU FIXTURE — no inference or scheduler evaluation" + (" — INCOMPLETE MATRIX" if missing else "")
    lines = ["# " + title, "", "These are per-seed descriptive results, not confidence intervals or SOTA claims.", "",
             "Streaming-event gaps are not token-level ITL. Failures remain in SLO denominators.", "",
             "| Policy | KV capacity | Seed | Phase | Completed / arrivals | SLO attainment | TTFT p99 (s) | Request max-event-gap p99 (s) | Flags |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    points = []
    def number(v):
        return "N/A" if v is None else f"{v:.4f}"
    for summary, meta in runs:
        labels = summary["labels"]
        total, passed = 0, 0
        for phase, data in summary["phases"].items():
            flags = summary["quality_flags"] + data["quality_flags"]
            lines.append(f"| {labels['policy']} | {labels['capacity']} | {labels['seed']} | {phase} | "
                         f"{data['completed']} / {data['arrivals']} | {number(data['slo_attainment_all_arrivals'])} | "
                         f"{number(data['ttft_p99_s'])} | {number(data['request_max_event_gap_p99_s'])} | {', '.join(flags)} |")
            total += data["arrivals"]
            passed += data["slo_passing_requests"]
        points.append({"labels": labels, "completed_tokens_per_drain_inclusive_s": summary["completed_output_tokens_per_drain_inclusive_s"],
                       "slo_attainment": passed/total if total else None})
    lines += ["", "## Sampled memory pressure", "",
              "90% is a descriptive threshold, not a causal or admission rule. Counter deltas use in-phase samples and omit unsampled boundaries.", "",
              "| Policy | KV capacity | Seed | Phase | Peak KV fraction | KV samples | Fraction of KV samples ≥90% | Observed preemption delta |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for summary, meta in runs:
        labels = summary["labels"]
        for phase, data in summary["phases"].items():
            lines.append(f"| {labels['policy']} | {labels['capacity']} | {labels['seed']} | {phase} | "
                         f"{number(data['kv_usage_peak_fraction'])} | {data['kv_usage_sample_count']} | "
                         f"{number(data['kv_samples_at_or_above_90pct_fraction'])} | "
                         f"{number(data['preemption_delta_between_in_phase_samples'])} |")
    lines += ["", "Missing arms: " + str(missing), "",
              "Inspect timelines, queue trends, GPU telemetry and server logs before assigning a cause to any difference."]
    (root/"comparison.md").write_text("\n".join(lines)+"\n")
    (root/"comparison.json").write_text(json.dumps({"title": title, "missing": missing, "points": points}, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for point, (summary, meta) in zip(points, runs):
        labels = point["labels"]
        label = labels["policy"] + "/" + labels["capacity"]
        color = {"fixed_2k": "tab:blue", "fixed_16k": "tab:orange", "ppas": "tab:green", "fixed_512": "tab:purple", "decode_guard": "tab:red", "reserve_static": "tab:cyan", "reserve_feedback": "tab:brown", "cap_2": "tab:pink", "evict_cost": "tab:olive", "evict_fair": "tab:gray"}[labels["policy"]]
        marker = "o" if labels["capacity"] == "roomy" else "x"
        if point["slo_attainment"] is not None:
            axes[0].scatter(point["completed_tokens_per_drain_inclusive_s"], point["slo_attainment"], c=color, marker=marker, label=label)
        for i, data in enumerate(summary["phases"].values()):
            if data["ttft_p99_s"] is not None:
                axes[1].scatter(i, data["ttft_p99_s"], c=color, marker=marker)
            if data["request_max_event_gap_p99_s"] is not None:
                axes[2].scatter(i, data["request_max_event_gap_p99_s"], c=color, marker=marker)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        axes[0].legend(unique.values(), unique.keys(), fontsize=7)
    axes[0].set_xlabel("Completed output tokens/s, including drain")
    axes[0].set_ylabel("Streaming SLO attainment, all arrivals")
    phases = list(runs[0][0]["phases"])
    for ax in axes[1:]:
        ax.set_xticks(range(len(phases)), phases)
    axes[1].set_ylabel("Exploratory request TTFT p99 (s)")
    axes[2].set_ylabel("Exploratory request max SSE-gap p99 (s)")
    fig.suptitle(title + " — each point is one seed")
    fig.tight_layout()
    fig.savefig(root/"comparison.png", dpi=150)
    plt.close(fig)
    print(root/"comparison.md")


if __name__ == "__main__":
    main()
