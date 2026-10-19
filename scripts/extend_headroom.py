"""Opt-in admission hooks for the pinned vLLM 0.22.1 scheduler."""
import ast


def extend(source):
    if '_lab_headroom' in source:
        raise ValueError('Headroom already applied')
    anchors = {
        '        self._pause_state: PauseState = PauseState.UNPAUSED': '''        self._pause_state: PauseState = PauseState.UNPAUSED
        from vllm.v1.core.sched.lab_headroom import attach
        self._lab_headroom = attach(self, os.getenv("KVLAB_HEADROOM", ""))''',
        '                num_external_computed_tokens = 0\n': '''                if self._lab_headroom is not None and not self._lab_headroom.allows(self, request):
                    break  # Keep FCFS head in place; never drop or bypass it.
                num_external_computed_tokens = 0
''',
        '        request.status = RequestStatus.PREEMPTED\n        request.num_computed_tokens = 0': '''        if self._lab_headroom is not None:
            self._lab_headroom.preempt()
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0''',
        '        return engine_core_outputs\n': '''        if self._lab_headroom is not None:
            self._lab_headroom.completed(engine_core_outputs)
        return engine_core_outputs
''',
    }
    for old, new in anchors.items():
        if source.count(old) != 1:
            raise ValueError('Missing or ambiguous headroom anchor: ' + old[:80])
        source = source.replace(old, new)
    ast.parse(source)
    return source


if __name__ == '__main__':
    import importlib.metadata
    import importlib.util
    from pathlib import Path
    import shutil
    if importlib.metadata.version('vllm') != '0.22.1':
        raise SystemExit('Headroom requires vLLM 0.22.1')
    root = Path(importlib.util.find_spec('vllm').origin).parent/'v1/core/sched'
    path = root/'scheduler.py'
    source = extend(path.read_text())
    shutil.copyfile(Path(__file__).with_name('headroom.py'), root/'lab_headroom.py')
    path.write_text(source)
