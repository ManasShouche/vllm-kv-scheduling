"""Fail-closed hooks for vLLM 0.22.1 after existing attributed patches."""
import ast


def instrument(source):
    anchors = {
        '        self._pause_state: PauseState = PauseState.UNPAUSED':
        '''        self._pause_state: PauseState = PauseState.UNPAUSED
        self._lab_trace = None
        if os.getenv("KVLAB_TRACE_DIR"):
            from vllm.v1.core.sched.lab_trace import Trace
            self._lab_trace = Trace(os.environ["KVLAB_TRACE_DIR"])''',
        '        with record_function_or_nullcontext("schedule: update_after_schedule"):\n':
        '''        if self._lab_trace is not None:
            self._lab_trace.scheduled(self, scheduler_output)
        with record_function_or_nullcontext("schedule: update_after_schedule"):
''',
        '        request.status = RequestStatus.PREEMPTED\n        request.num_computed_tokens = 0':
        '''        if self._lab_trace is not None:
            self._lab_trace.preempt(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0''',
        '        return engine_core_outputs\n':
        '''        if self._lab_trace is not None:
            self._lab_trace.completed(scheduler_output, engine_core_outputs)
        return engine_core_outputs
''',
    }
    if '_lab_trace' in source:
        raise ValueError('Recompute instrumentation already applied')
    for old, new in anchors.items():
        if source.count(old) != 1:
            raise ValueError('Recompute hook missing or ambiguous: '+old[:80])
        source = source.replace(old, new)
    ast.parse(source)
    return source


if __name__ == '__main__':
    import importlib.util
    import importlib.metadata
    from pathlib import Path
    import shutil
    if importlib.metadata.version('vllm') != '0.22.1':
        raise SystemExit('Recompute hooks require exactly vLLM 0.22.1')
    root = Path(importlib.util.find_spec('vllm').origin).parent/'v1/core/sched'
    scheduler = root/'scheduler.py'
    patched = instrument(scheduler.read_text())
    shutil.copyfile(Path(__file__).with_name('scheduler_trace.py'), root/'lab_trace.py')
    scheduler.write_text(patched)
