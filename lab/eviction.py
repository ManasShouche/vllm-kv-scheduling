"""Experimental victim selection, not a reproduction of UniBoost or a cost model."""


def victim_index(running, start, blocks, fair=False):
    """Only unscheduled suffix is eligible; preserve upstream rollback invariants.

Computed positions proxy lost work, not measured GPU milliseconds. Repeat
penalty discourages concentrating evictions. Ties keep upstream newest victim.
A 5% margin avoids changing victim for nearly identical estimated costs.
"""
    if not 0 <= start < len(running):
        raise ValueError('Invalid scheduling cursor')
    candidates = [i for i in range(start, len(running)) if blocks(running[i]) > 0]
    if not candidates:
        return len(running)-1
    def cost(i):
        r = running[i]
        return r.num_computed_tokens * (1+r.num_preemptions if fair else 1)
    tail = len(running)-1
    best = min(candidates, key=lambda i: (cost(i), -i))
    if tail in candidates and cost(best) >= .95*cost(tail):
        return tail
    return best


class Eviction:
    def __init__(self, mode):
        if mode not in ('evict_cost', 'evict_fair'):
            raise ValueError('Unknown eviction policy')
        self.mode = mode
        self.seen = set()
        self.decisions = 0
        self.mark('init')

    def mark(self, event):
        if event not in self.seen:
            print(f'KVLAB_EVICTION event={event} mode={self.mode}', flush=True)
            self.seen.add(event)

    def choose(self, scheduler, start):
        self.mark('pressure')
        index = victim_index(scheduler.running, start,
            lambda r: len(scheduler.kv_cache_manager.get_blocks(r.request_id).blocks[0]),
            fair=self.mode == 'evict_fair')
        # Bounded structured diagnostics: no prompt text or token IDs.
        import json
        self.decisions += 1
        if self.decisions <= 256:
            rows = []
            for i, request in enumerate(scheduler.running):
                blocks = len(scheduler.kv_cache_manager.get_blocks(request.request_id).blocks[0])
                rows.append(dict(index=i, request_id=request.request_id,
                    computed=request.num_computed_tokens, preemptions=request.num_preemptions,
                    blocks=blocks, eligible=i >= start and blocks > 0,
                    cost=request.num_computed_tokens * (1+request.num_preemptions if self.mode == 'evict_fair' else 1)))
            print('KVLAB_EVICTION_DECISION '+json.dumps(dict(
                sequence=self.decisions, mode=self.mode, cursor=start, selected=index,
                default=len(rows)-1, requests=rows), separators=(',', ':')), flush=True)
        elif self.decisions == 257:
            self.mark('diagnostics_truncated')
        if index != len(scheduler.running)-1:
            self.mark('changed')
        return index


def attach(scheduler, mode):
    if not mode:
        return None
    # Reuse the same conservative supported-runtime check, without retaining
    # an admission controller or emitting its activation markers.
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    groups = scheduler.kv_cache_config.kv_cache_groups
    if (len(groups) != 1 or type(groups[0].kv_cache_spec) is not FullAttentionSpec
            or scheduler.scheduler_config.async_scheduling
            or scheduler.cache_config.enable_prefix_caching
            or scheduler.connector is not None or scheduler.is_encoder_decoder
            or scheduler.num_lookahead_tokens != 0
            or scheduler.dcp_world_size != 1 or scheduler.pcp_world_size != 1
            or scheduler.parallel_config.tensor_parallel_size != 1
            or scheduler.parallel_config.pipeline_parallel_size != 1
            or scheduler.ppas_enabled or scheduler.max_num_scheduled_tokens != 512
            or scheduler.scheduler_config.policy != 'fcfs'
            or scheduler._lab_headroom is not None):
        raise ValueError('Eviction experiment requires single-GPU full-attention sync FCFS fixed512 without admission/prefix/speculation')
    return Eviction(mode)
