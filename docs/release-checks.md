# Public-copy verification

Prepared 19 October 2026 as a separate copy; original development tree preserved.

- Python 3.12 suite: 86 tests discovered, 84 passed and 2 skipped because the optional
  full upstream scheduler checkout is not distributed in this public copy.
- Six-arm CPU HTTP/SSE self-check passed, including collection, audits and comparison.
- Vendored P-PAS content hash and original policy/config lock hashes verified.
- Relative Markdown links checked.
- Checked distributable text for the original account number, local user paths,
  AWS resource IDs, access-key patterns, private-key blocks and S3 bucket URLs;
  no matches. This is a targeted check, not a guarantee against every possible secret.
- No raw cloud metadata, private operational history, virtualenv, caches or raw
  archives are included in this copy.

CI configuration is supplied as `docs/ci-workflow.yml`, but is not installed or running on GitHub. The authenticated CLI token lacks the workflow scope. To enable Actions later, place this template at `.github/workflows/tests.yml` using an account session with workflow permission. No new GPU experiment
was executed during repository cleanup. See README.md for the CPU-only self-check.
