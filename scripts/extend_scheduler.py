"""Opt-in decode-sensitive prefill cap, applied AFTER the unmodified vendor patch."""
import ast


def extend(source):
    changes = {
        '        self.ppas_enabled = os.getenv("PPAS_ENABLED", "0") == "1"':
        '''        self.ppas_enabled = os.getenv("PPAS_ENABLED", "0") == "1"
        self.lab_decode_guard = os.getenv("KVLAB_DECODE_GUARD", "0") == "1"
        if self.lab_decode_guard and not self.ppas_enabled:
            raise ValueError("Decode guard requires the adaptive scheduling path")
        if self.lab_decode_guard:
            print("KVLAB_VARIANT name=decode_guard", flush=True)''',
        'num_active_prefills >= 2': 'num_active_prefills >= (1 if self.lab_decode_guard else 2)',
        '                2048\n                if under_prefill_pressure':
        '                (512 if self.lab_decode_guard else 2048)\n                if under_prefill_pressure',
    }
    for old, new in changes.items():
        if source.count(old) != 1:
            raise ValueError("Decode-guard anchor missing or ambiguous")
        source = source.replace(old, new)
    ast.parse(source)
    return source


if __name__ == '__main__':
    import importlib.util
    from pathlib import Path
    root = Path(importlib.util.find_spec('vllm').origin).parent
    scheduler = root/'v1/core/sched/scheduler.py'
    scheduler.write_text(extend(scheduler.read_text()))
