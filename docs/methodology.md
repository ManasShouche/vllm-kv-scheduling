# Measurement and limitations

Requests use exact-length synthetic token prompts and a frozen open-loop arrival
schedule, independently of completions. Each arm starts a fresh server. The held-out
comparison uses seeds 201/202/203, prompt lengths 2048/8192/12288, short/long output
phases of 64/2048 tokens, 1 GiB KV cache and synchronous scheduling with tracing.
Model revision: `aa8e72537993ba99e69dfaafa59ed015b17504d1`.

The cost policy selects from the unscheduled suffix of running requests that hold
releasable blocks. It prefers lower computed-position cost and keeps the default
victim when savings are below 5%. This is an experimental heuristic, not an output-
length oracle. Candidate eligibility can eliminate any opportunity to change victims.

- TTFT in the frozen guards is actual-dispatch-to-first-text-event. Retrospective
  research analysis also reports intended-arrival-relative latency; they differ.
- Streaming gaps measure nonempty SSE text events, not token-level inter-token latency.
- Recomputed positions count repeated scheduler work, not measured CUDA kernel time.
- Failed and censored requests remain in SLO denominators. Warmup exclusions are
  fixed by arrival time. Completion throughput includes drain time.
- Each seed must independently pass the original thresholds. We do not pool away
  failed seeds. Zero baseline recomputation is uninformative, not a mechanism win.
- Small samples make p95/p99 exploratory. The experiment uses one GPU/model and
  synthetic workloads; production generalization and significance are unestablished.
- The P-PAS patch changes the running-request context clamp. That upstream behavior
  is preserved for reproduction, not endorsed for production. See the patch itself.
- Tracing overhead was screened in an earlier ABBA experiment; decision logging's
  overhead in the latest candidate is included and not independently isolated.

The overall result is mixed. Aggregate guards can hide individual regressions;
public paired completion records retain that evidence. No accepted paper, SOTA
claim, or general superiority is implied by this repository.
