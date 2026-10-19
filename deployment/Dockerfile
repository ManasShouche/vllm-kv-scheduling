ARG VLLM_BASE
FROM ${VLLM_BASE}
USER root
RUN if ! command -v git >/dev/null; then apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*; fi
COPY vendor/ppas_vllm_0.22.1.patch /opt/reproduction/ppas.patch
COPY vendor/provenance.json /opt/reproduction/provenance.json
COPY scripts/apply_patch.py /opt/reproduction/apply_patch.py
RUN python3 /opt/reproduction/apply_patch.py
COPY scripts/extend_scheduler.py /opt/reproduction/extend_scheduler.py
RUN python3 /opt/reproduction/extend_scheduler.py
COPY scripts/instrument_scheduler.py /opt/reproduction/instrument_scheduler.py
RUN python3 /opt/reproduction/instrument_scheduler.py
COPY lab/scheduler_trace.py /opt/reproduction/scheduler_trace.py
COPY scripts/instrument_recompute.py /opt/reproduction/instrument_recompute.py
RUN python3 /opt/reproduction/instrument_recompute.py
COPY lab/headroom.py /opt/reproduction/headroom.py
COPY scripts/extend_headroom.py /opt/reproduction/extend_headroom.py
RUN python3 /opt/reproduction/extend_headroom.py
COPY lab/eviction.py /opt/reproduction/eviction.py
COPY scripts/extend_eviction.py /opt/reproduction/extend_eviction.py
RUN python3 /opt/reproduction/extend_eviction.py
# Use the official image's original vLLM serving entrypoint.
