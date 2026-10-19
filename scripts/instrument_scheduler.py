"""Add one-shot runtime observations separately from the unmodified vendor patch."""
import ast


def instrument(source):
    anchors = {
        '        self.ppas_enabled = os.getenv("PPAS_ENABLED", "0") == "1"':
        '''        self.ppas_enabled = os.getenv("PPAS_ENABLED", "0") == "1"
        self._lab_pressure_seen = False
        self._lab_step_seen = False
        print(f"KVLAB_PPAS_INIT enabled={int(self.ppas_enabled)}", flush=True)''',
        '        if self.ppas_enabled:\n            num_prefill_running = sum(':
        '''        if self.ppas_enabled:
            if not self._lab_step_seen:
                print("KVLAB_PPAS_STEP enabled=1", flush=True)
                self._lab_step_seen = True
            num_prefill_running = sum(''',
        '            # Tracks aggregate prefill tokens admitted in this scheduler step.':
        '''            if under_prefill_pressure and not self._lab_pressure_seen:
                print(f"KVLAB_PPAS_PRESSURE active_prefills={num_active_prefills} decodes={num_running_decodes} budget={ppas_prefill_budget}", flush=True)
                self._lab_pressure_seen = True
            # Tracks aggregate prefill tokens admitted in this scheduler step.''',
    }
    for old, new in anchors.items():
        if source.count(old) != 1:
            raise ValueError("Scheduler observation anchor missing or ambiguous")
        source = source.replace(old, new)
    ast.parse(source)
    return source


if __name__ == '__main__':
    import importlib.util
    from pathlib import Path
    root = Path(importlib.util.find_spec('vllm').origin).parent
    path = root/'v1/core/sched/scheduler.py'
    path.write_text(instrument(path.read_text()))
