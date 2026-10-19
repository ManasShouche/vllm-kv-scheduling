"""Opt-in host-clock observations; no prompts, token IDs or GPU-time claims."""
import atexit
import json
import os
from pathlib import Path
import time


class Trace:
    def __init__(self, directory):
        self.path = Path(directory)/f'trace-{os.getpid()}.jsonl'
        self.file = self.path.open('x', buffering=1024*1024)
        self.sequence = 0
        self.step = 0
        self.bytes = 0
        self.pending = {}
        self.high_water = {}
        self.emit('header', clock='perf_counter_ns_same_linux_host', schema=1)
        self.file.flush()
        atexit.register(self.close)

    def emit(self, kind, **fields):
        self.sequence += 1
        line = json.dumps(dict(event=kind, sequence=self.sequence,
                               time_ns=time.perf_counter_ns(), **fields), separators=(',', ':'))+'\n'
        self.bytes += len(line.encode())
        if self.bytes > 256*1024*1024:
            raise RuntimeError('Telemetry exceeded 256 MiB; refusing silent loss')
        self.file.write(line)

    def preempt(self, request):
        self.emit('preempt', request_id=request.request_id,
                  computed_before_reset=request.num_computed_tokens,
                  emitted_tokens=request.num_output_tokens,
                  prior_preemptions=request.num_preemptions)
        self.file.flush()

    def scheduled(self, scheduler, output):
        if not output.num_scheduled_tokens:
            return
        if id(output) in self.pending:
            raise RuntimeError('Duplicate outstanding scheduler output')
        self.step += 1
        rows = []
        for rid, count in output.num_scheduled_tokens.items():
            request = scheduler.requests[rid]
            start = request.num_computed_tokens
            end = start + count
            repeated = max(0, min(end, self.high_water.get(rid, 0))-start)
            rows.append(dict(request_id=rid, computed_before=start, scheduled_tokens=count,
                             prompt_tokens=request.num_prompt_tokens,
                             emitted_tokens=request.num_output_tokens,
                             preemptions=request.num_preemptions,
                             recompute_scheduled_tokens=repeated))
        self.pending[id(output)] = (self.step, rows)
        self.emit('scheduled', step=self.step, requests=rows,
                  free_kv_blocks=scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
                  running=len(scheduler.running), waiting=len(scheduler.waiting))

    def completed(self, output, engine_outputs):
        # Hook runs after update_from_output has consumed model results. This is
        # CPU result availability, not CUDA event timing or network emission.
        if not output.num_scheduled_tokens:
            return
        step, rows = self.pending.pop(id(output))
        for row in rows:
            rid = row['request_id']
            self.high_water[rid] = max(self.high_water.get(rid, 0),
                                       row['computed_before']+row['scheduled_tokens'])
        emitted = []
        for group in engine_outputs.values():
            for item in group.outputs:
                finished = item.finish_reason is not None
                emitted.append(dict(request_id=item.request_id,
                                    new_tokens=len(item.new_token_ids), finished=finished))
                if finished:
                    self.high_water.pop(item.request_id, None)
        self.emit('completed', step=step, outputs=emitted)
        if step % 128 == 0 or any(row['finished'] for row in emitted):
            self.file.flush()

    def close(self):
        if not self.file.closed:
            self.emit('footer', pending_steps=len(self.pending))
            self.file.close()
