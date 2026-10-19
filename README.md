# KV Pressure Lab

A reproducible systems research project studying **KV-cache pressure, eviction cost,
and request latency in vLLM**. It combines an open-loop inference benchmark,
request-level scheduler traces, experimental policies, and held-out evaluation.

**Status:** research prototype and completed experiment series; not a production
scheduler or a demonstrated state-of-the-art result.

## Main result

On an NVIDIA A10G with vLLM 0.22.1 and Qwen2.5-3B-Instruct, choosing a lower-cost
eligible eviction victim reduced repeated token-position scheduling on two of
three held-out workloads. Aggregate latency and throughput guardrails passed on
all three; the complete predefined screen did **not** pass.

| Held-out seed | Recomputed positions: baseline → candidate | Reduction | Screen |
|---|---:|---:|---|
| 201 | 12,877 → 2,988 | 76.80% | Pass |
| 202 | 8,740 → 2,737 | 68.68% | Pass |
| 203 | 19,233 → 19,232 | 0.01% | Fail: no changed victim |

Throughput was essentially unchanged. Some individual requests completed up to
**19.69 seconds later** under the candidate. This is evidence of a workload-dependent
tradeoff, not a general speedup. Three seeds do not establish statistical significance.

- [Results and fairness](docs/heldout-results.md)
- [Frozen evaluation plan](docs/heldout-plan.md)
- [Earlier experiments, including negative results](docs/experiment-history.md)
- [Measurement contract and limitations](docs/methodology.md)

## Run locally — no AWS or GPU needed

Use Python **3.12**. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-client.txt
python -m unittest discover -s tests -v
python scripts/summarize_public_results.py
python -m lab.selfcheck --output /tmp/kv-pressure-fixture-unique
```

Choose a new output directory for each self-check. It runs a local HTTP/SSE
fixture, audits six synthetic arms, and generates reports. **It performs no model
inference and its timings are not research results.** Tests needing an optional
upstream scheduler checkout are skipped when that checkout is absent.

The summary command recomputes the displayed reductions and reports paired
completion differences from the included aggregate records. It is not an
independent raw-trace reproduction. See [data access](docs/data-access.md).

## Code map

| Path | Purpose |
|---|---|
| `lab/workload.py`, `collect.py` | Frozen synthetic schedules and open-loop SSE collection |
| `lab/eviction.py` | Cost-aware and repeat-penalized victim selection |
| `lab/scheduler_trace.py`, `trace_audit.py` | Request identity, repeated positions, and lifecycle checks |
| `lab/audit.py`, `analyze.py`, `compare.py` | Artifact validation and matched comparisons |
| `lab/headroom.py`, `feasibility.py` | Earlier controller experiments; feasibility kernel is CPU-only |
| `scripts/` | Pinned patching, dedicated-host execution, historical audit tools |
| `configs/` | Experimental configurations; held-out lock retains original hashes |
| `results/heldout/` | Public numerical summaries and paired fairness records |
| `vendor/` | Unmodified P-PAS patch, license, and upstream provenance |

[GPU reproduction](docs/reproduction.md) requires a separately provisioned,
dedicated Linux GPU host. Nothing in the local quickstart launches cloud resources.
Historical scripts are tied to named experiment artifacts; do not mistake them
for tests that work without the corresponding raw archives.

## Attribution

The original comparison uses the [P-PAS implementation](https://github.com/TimoSaemann/ppas-vllm),
vendored unmodified at commit `f36197dddf485653f7eb8c89be96fd2d2ebca2a2`.
Our experimental extensions are applied separately. Upstream attribution and
hashes are in `vendor/provenance.json` and `NOTICE`. The inference engine is
[vLLM](https://github.com/vllm-project/vllm); model weights are obtained separately.
New project code uses the included Apache-2.0 license. Strong published-baseline
reproduction and novelty assessment remain unfinished.
