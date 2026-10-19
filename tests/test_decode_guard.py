import contextlib
import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.extend_scheduler import extend
from scripts.instrument_scheduler import instrument
from lab.activation import assess
from lab.workload import validate

FIXTURE = Path(__file__).parent/'fixtures/ppas_scheduler_fragment.py'


def scheduler_class(source):
    scope = {}
    exec(compile(source, str(FIXTURE), 'exec'), scope)
    return scope['Scheduler']


def populate(scheduler, prefills, decodes):
    scheduler.running = [SimpleNamespace(num_computed_tokens=8193, num_prompt_tokens=8192) for _ in range(decodes)]
    scheduler.waiting = [SimpleNamespace(num_computed_tokens=0, num_prompt_tokens=8192) for _ in range(prefills)]


class DecodeGuardTests(unittest.TestCase):
    def test_disabled_extension_preserves_upstream_budget_decisions(self):
        original = scheduler_class(FIXTURE.read_text())
        extended = scheduler_class(extend(FIXTURE.read_text()))
        with patch.dict('os.environ', {'PPAS_ENABLED': '1', 'KVLAB_DECODE_GUARD': '0'}):
            for prefills in range(4):
                for decodes in range(4):
                    a, b = original(), extended()
                    populate(a, prefills, decodes); populate(b, prefills, decodes)
                    self.assertEqual(a.schedule(), b.schedule())

    def test_guard_caps_single_prefill_only_while_decode_active(self):
        cls = scheduler_class(instrument(extend(FIXTURE.read_text())))
        out = io.StringIO()
        with patch.dict('os.environ', {'PPAS_ENABLED': '1', 'KVLAB_DECODE_GUARD': '1'}), contextlib.redirect_stdout(out):
            scheduler = cls()
            populate(scheduler, 1, 0)
            self.assertEqual(scheduler.schedule(), 16384)
            populate(scheduler, 1, 1)
            self.assertEqual(scheduler.schedule(), 512)
            self.assertEqual(scheduler.schedule(), 512)
            populate(scheduler, 0, 1)
            self.assertEqual(scheduler.schedule(), 16384)
        log = out.getvalue()
        self.assertEqual(log.count('KVLAB_PPAS_PRESSURE'), 1)
        self.assertTrue(assess(log, 'decode_guard')['comparison_ready'])
        self.assertFalse(assess(log, 'ppas')['activation_verified'])
        self.assertFalse(assess(log, 'fixed_512')['activation_verified'])

    def test_invalid_activation_and_repeat_patching_fail_closed(self):
        cls = scheduler_class(extend(FIXTURE.read_text()))
        with patch.dict('os.environ', {'PPAS_ENABLED': '0', 'KVLAB_DECODE_GUARD': '1'}), self.assertRaises(ValueError):
            cls()
        with self.assertRaises(ValueError):
            extend(extend(FIXTURE.read_text()))
        self.assertFalse(assess('KVLAB_PPAS_INIT enabled=1\nKVLAB_PPAS_STEP enabled=1', 'decode_guard')['activation_verified'])

    def test_configuration_pins_policy_budgets(self):
        config=json.loads(Path('configs/improvement-screen.json').read_text())
        validate(config)
        config['policies']['fixed_512']=2048
        with self.assertRaises(ValueError):
            validate(config)
