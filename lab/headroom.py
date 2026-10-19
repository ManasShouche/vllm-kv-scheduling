"""Experimental admission control; single full-attention KV group only.

Reserve unfinished known context plus a bounded future decode horizon. This is
an admission heuristic, not a deadline guarantee or an output-length predictor.
"""


class Headroom:
    def __init__(self, mode):
        if mode not in ('reserve_static', 'reserve_feedback'):
            raise ValueError('Unknown headroom mode')
        self.mode = mode
        self.horizon = 256
        self.progress = 0
        self.seen = set()

    def mark(self, event):
        if event not in self.seen:
            print(f'KVLAB_HEADROOM event={event} mode={self.mode}', flush=True)
            self.seen.add(event)

    def preempt(self):
        self.mark('preempt')
        if self.mode == 'reserve_feedback':
            self.horizon = min(2048, self.horizon * 2)
            self.progress = 0

    def completed(self, outputs):
        if self.mode != 'reserve_feedback':
            return
        self.progress += sum(len(row.new_token_ids)
                             for batch in outputs.values() for row in batch.outputs)
        # Decay only with useful output progress, never idle scheduler polling.
        while self.progress >= 4096:
            self.progress -= 4096
            self.horizon = max(256, self.horizon - 128)

    def allows(self, scheduler, candidate):
        self.mark('step')
        if not scheduler.running:
            # Liveness: drain head-of-line even if its growth reserve cannot fit.
            # The ordinary vLLM allocator still decides physical feasibility.
            return True
        manager = scheduler.kv_cache_manager
        block_size = scheduler.block_size
        required = 0
        for request in [*scheduler.running, candidate]:
            blocks = manager.get_blocks(request.request_id).blocks
            if len(blocks) != 1:
                raise ValueError('Headroom requires one KV cache group')
            # Actual allocations already include this step's running chunks.
            # Reserve all remaining prompt/recompute work, not just next chunk.
            target = min(scheduler.max_model_len, request.num_tokens + self.horizon)
            required += max(0, (target + block_size - 1) // block_size - len(blocks[0]))
        allowed = required <= manager.block_pool.get_num_free_blocks()
        if not allowed:
            self.mark('defer')
        return allowed


def attach(scheduler, mode):
    if not mode:
        return None
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    groups = scheduler.kv_cache_config.kv_cache_groups
    if (len(groups) != 1 or type(groups[0].kv_cache_spec) is not FullAttentionSpec
            or scheduler.cache_config.enable_prefix_caching
            or scheduler.scheduler_config.async_scheduling
            or scheduler.connector is not None or scheduler.is_encoder_decoder
            or scheduler.num_lookahead_tokens != 0
            or scheduler.dcp_world_size != 1 or scheduler.pcp_world_size != 1
            or scheduler.parallel_config.tensor_parallel_size != 1
            or scheduler.parallel_config.pipeline_parallel_size != 1
            or scheduler.ppas_enabled or scheduler.max_num_scheduled_tokens != 512):
        raise ValueError('Headroom supports only sync single-GPU full-attention fixed512 without prefix caching or speculation')
    controller = Headroom(mode)
    controller.mark('init')
    return controller
