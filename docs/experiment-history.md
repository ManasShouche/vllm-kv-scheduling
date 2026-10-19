# Experiment history

These are completed project experiments, not independent published findings.
Raw run archives remain preserved separately; dates are experiment identifiers.

| Run | Finding |
|---|---|
| Initial pilot, 2026-10-06 | 186/226 requests completed; overload and 40 timeouts retained |
| Lower-load calibration, 2026-10-07 | Collection validated; no scheduler superiority established |
| Seed 43 replication | Fixed-512 improved streaming SLO but TTFT p95 rose 21.85%, failing the 10% guard |
| Telemetry, 2026-10-08 | 244/244 completed; trace identity checks and ABBA overhead screen passed |
| Headroom, 2026-10-09 | Feedback reduced recomputation 53.24%, but TTFT worsened 63% and throughput fell 8.1%; failed |
| Eviction first screen | Cap-2 produced four timeouts; sweep stopped before candidate arms |
| Eviction retry | 183/183 completed, but neither candidate changed a victim; mechanism unexercised |
| Mixed seed 102, 2026-10-13 | Cost policy reduced recomputation 42.95%; below the frozen 50% threshold; repeat preemption observed |
| Held-out seeds 201/202/203, 2026-10-14 | Two seeds passed; third had no changed victim; full screen failed; individual completion regressions remain |

The next useful research step is explaining the request-level regressions and
eligibility limitations from existing traces. Additional compute is not evidence
of novelty. The original prototype histories and private operational context have
not been rewritten to make the final result look better.
