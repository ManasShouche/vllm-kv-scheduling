"""Fail-closed pinned runtime hook; preserves already-scheduled request state."""
import ast


def extend(source):
    if '_lab_eviction' in source:
        raise ValueError('Eviction hook already applied')
    changes = {
        '        self._lab_headroom = attach(self, os.getenv("KVLAB_HEADROOM", ""))': '''        self._lab_headroom = attach(self, os.getenv("KVLAB_HEADROOM", ""))
        from vllm.v1.core.sched.lab_eviction import attach as attach_eviction
        self._lab_eviction = attach_eviction(self, os.getenv("KVLAB_EVICTION", ""))''',
        '                        preempted_req = self.running.pop()': '''                        victim_index = (self._lab_eviction.choose(self, req_index)
                                        if self._lab_eviction is not None else -1)
                        preempted_req = self.running.pop(victim_index)''',
    }
    for old,new in changes.items():
        if source.count(old) != 1:
            raise ValueError('Eviction anchor missing or ambiguous')
        source=source.replace(old,new)
    ast.parse(source)
    return source


if __name__ == '__main__':
    import importlib.metadata
    import importlib.util
    import shutil
    from pathlib import Path
    if importlib.metadata.version('vllm') != '0.22.1':
        raise SystemExit('Eviction hook requires vLLM0.22.1')
    root=Path(importlib.util.find_spec('vllm').origin).parent/'v1/core/sched'
    p=root/'scheduler.py';source=extend(p.read_text())
    shutil.copyfile(Path(__file__).with_name('eviction.py'),root/'lab_eviction.py')
    p.write_text(source)
