# Frozen cost-policy validation — 14 October 2026

Compare unchanged evict_cost against fixed512 on new seeds201,202,203. Six
fresh-server arms on one A10G, same pinned image/model,1GiB KV, synchronous
scheduling/tracing, prompt mixture2048/8192/12288 and original phase lengths/load.
No output results from these seeds were used for policy selection. Policy hashes,
configuration hash and deterministic randomized order are in cost-heldout-lock.json.
No algorithm or threshold changes are authorized within this validation batch.
Development seed102 failed its50%mechanism threshold; this remains unchanged.

For EACH seed require both phases TTFT p95<=1.10x baseline, median worst SSE gap
<=1.10x, throughput>=.95x, no extra failed requests and no lower joint SLO fraction.
Require>=50%recomputation reduction when baseline recomputation is nonzero;
if zero, mark mechanism comparison uninformative, not a win. Also require actual
changed-victim evidence for intervention readiness. Report every seed separately;
do not pool away a regression or replace any seed after viewing results.

Fairness diagnostics: match request IDs across policies; report increase in
individual TTFT and arrival-to-completion time, median/p95/max paired differences,
max per-request preemptions and count with>=2evictions; break down by phase and
prompt length. Mark timeout durations as censored and retain failures in all
SLO denominators. These fairness diagnostics are descriptive, not a new post-hoc
pass threshold. Instrumentation overhead is included and remains unisolated.
With only three seeds, use cautious descriptive conclusions, not SOTA or broad
statistical claims. Candidate validation does not substitute for published
strong-baseline reproduction or novelty assessment.

