# Extracted from pinned PPAS scheduler f36197dddf485653f7eb8c89be96fd2d2ebca2a2.
# Only initialization and budget decision are isolated; no GPU dependencies.
import os
import itertools
class Scheduler:
    def __init__(self):
        self.ppas_enabled = os.getenv("PPAS_ENABLED", "0") == "1"
        self.running = []
        self.waiting = []
        self.skipped_waiting = []
        self.max_num_scheduled_tokens = 16384
    def schedule(self):
        if self.ppas_enabled:
            num_prefill_running = sum(
                request.num_computed_tokens < request.num_prompt_tokens
                for request in self.running
            )

            num_running_decodes = len(self.running) - num_prefill_running

            num_prefill_waiting = sum(
                request.num_computed_tokens < request.num_prompt_tokens
                for request in itertools.chain(
                    self.waiting,
                    self.skipped_waiting,
                )
            )

            num_active_prefills = (
                    num_prefill_running
                    + num_prefill_waiting
            )

            under_prefill_pressure = (
                    num_active_prefills >= 2
                    and num_running_decodes > 0
            )

            ppas_prefill_budget = (
                2048
                if under_prefill_pressure
                else self.max_num_scheduled_tokens
            )

            # Tracks aggregate prefill tokens admitted in this scheduler step.
            prefill_tokens_scheduled = 0
        return locals().get('ppas_prefill_budget')
