# Held-out validation — 14 October 2026

Six frozen arms completed successfully on one A10G: unchanged evict_cost versus fixed_512, seeds 201/202/203. Archive checksum verified; frozen policy/config hashes and arm order verified; all six artifact and request-trace audits passed. No seed replaced and no policy retuned.

| Seed | Baseline recomputed positions | Candidate | Reduction | Frozen screen |
|---|---:|---:|---:|---|
| 201 | 12877 | 2988 | 76.80% | PASS |
| 202 | 8740 | 2737 | 68.68% | PASS |
| 203 | 19233 | 19232 | 0.01% | FAIL |

All seeds passed the latency, streaming-gap, throughput, failure-count and joint-SLO guardrails. Seed 203 did not change a victim and failed the 50% mechanism threshold. Therefore the full per-seed validation did NOT pass. Throughput was essentially unchanged; reduced recomputation is not an observed throughput improvement. Results support a workload-dependent mechanism, not SOTA or broad scheduler superiority.

## Request-level fairness

All 149 non-warmup matched request pairs completed; no censored pairs. Neither policy repeatedly preempted any request in this batch (maximum one per request). However individual completion times can worsen despite aggregate guardrails:

| Seed | Median completion change | P95 change | Maximum increase |
|---|---:|---:|---:|
| 201 | 0.0010s | 0.0603s | 19.688s |
| 202 | -0.0031s | 0.1091s | 8.924s |
| 203 | -0.0083s | 0.0084s | 0.071s |

Positive changes mean slower with candidate; completion is arrival-to-finish. TTFT guards and paired TTFT use actual dispatch-to-first-text-event. Per-phase/prompt-length diagnostics and all paired observations are retained in independent-fairness.json. SSE events are not tokens, and traced repeated positions are not CUDA execution time. Three seeds, sparse tails, no significance claim and no reproduced strong published scheduler baseline.

## Next research step

Explain the 19.69s and 8.92s individual slowdowns and seed 203's lack of eligible alternative victims from saved traces before more compute. Consider a cost-aware rule constrained by request age or accumulated service delay, with a separately frozen evaluation. Do not silently retune this completed validation or substitute seeds. Current evidence is useful for an honest research project/SOP; publication novelty and strong-baseline comparisons remain unfinished.


Raw archive SHA256: `43cbdaaf4aa3071110293171de81f958517810f85526c2e5e8b78dadad553ee1`. See [data access](data-access.md).
