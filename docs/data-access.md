# Data and provenance

This repository includes held-out arm summaries, frozen gate calculations, and all
149 non-warmup paired fairness records. Cloud account/instance identifiers, host
logs, personal paths, billing and local handoff records are excluded.

Raw artifacts and the original development snapshot are preserved in private S3
storage. This repository does **not** provide anonymous access or claim that a
reader can independently reconstruct all results from the public summaries alone.
The project owner must arrange raw-artifact access or a separately reviewed data
release. Credentials must never be committed to the repository.

For the held-out run, the original `results.tar.gz` SHA256 is:

`43cbdaaf4aa3071110293171de81f958517810f85526c2e5e8b78dadad553ee1`

After obtaining that archive and its `.sha256` file, place both under
`evidence/heldout-20261014a/`. Verify the checksum before extraction, and extract
with Python's `tarfile` data filter into that directory's `extracted/` folder.
Then run from the repository root:

```bash
PYTHONPATH=. python scripts/audit_heldout_snapshot.py
```

It validates frozen source/config hashes, complete matched arms, trace integrity,
per-seed thresholds, and paired fairness. Outputs are written under `evidence/`.
The original archive layout is `extracted/kv-pressure-run/kv-pressure-lab/`.
The other `audit_*_snapshot.py` tools similarly require their named historical
archives; `scripts/reproduce_research.py` uses three earlier matched archives.

The public report is derived from `independent-gates.json`,
`independent-fairness.json`, and `arm-summaries.json` in `results/heldout/`.
Those are derived records, not substitutes for the raw trace audit.
