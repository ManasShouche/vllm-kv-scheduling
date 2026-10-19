# GPU reproduction

The historical environment was a dedicated Linux NVIDIA A10G host, vLLM 0.22.1,
Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, and a
1 GiB KV cache. The configuration and policy hashes are frozen in
`configs/cost-heldout-lock.json`. Do not edit these and relabel a run as the
original held-out experiment.

The runtime helpers currently require EC2 identity; arbitrary GPU hosts require
an explicit portability change. Install working NVIDIA drivers, Docker, NVIDIA
Container Toolkit and the Python client dependencies. `scripts/build_on_ec2.sh`
resolves the official image tag to an immutable digest and records the resulting
image, dependencies and flags. A new build is not guaranteed bit-identical to the
historical image; retain its provenance and treat it as a new reproduction.

On the dedicated host, from the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-client.txt
bash scripts/build_on_ec2.sh
RUN_ID=heldout-reproduction PILOT_MODE=heldout \
PILOT_CONFIG=configs/cost-heldout.json \
MODEL_REVISION=aa8e72537993ba99e69dfaafa59ed015b17504d1 \
bash scripts/run_first_arm_on_ec2.sh
```

Use a unique run ID for every execution. This command runs six arms and packages
results; **it does not provision or automatically shut down the host**. Arrange a
separate verified runtime deadline before executing and preserve results before
terminating the host. Stopped EC2 instances can retain billable disks.

Optional AWS bootstrap and supervisor tools are included for dedicated hosts.
Read their prerequisites before use: they can stop the host, require systemd/IAM
permissions, and do not impose an account-wide spending cap. Configure your own
private bucket/role and verify shutdown behavior; no account-specific launch
configuration is distributed. Never run these on a shared machine.

`lab.compare` and `lab.trace_audit` audit newly generated workload directories.
The historical snapshot audit intentionally uses the original run ID and archive
layout; it is for verifying original evidence, not discovering arbitrary new runs.
