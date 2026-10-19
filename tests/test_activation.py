import unittest
from lab.activation import assess
from scripts.instrument_scheduler import instrument


class ActivationTests(unittest.TestCase):
    def test_environment_alone_is_not_activation(self):
        self.assertFalse(assess('PPAS_ENABLED=1', 'ppas')['activation_verified'])

    def test_enabled_step_without_pressure_is_not_comparison_ready(self):
        result = assess('KVLAB_PPAS_INIT enabled=1\nKVLAB_PPAS_STEP enabled=1', 'ppas')
        self.assertTrue(result['activation_verified'])
        self.assertFalse(result['comparison_ready'])

    def test_pressure_and_conflicting_markers(self):
        log = 'KVLAB_PPAS_INIT enabled=1\nKVLAB_PPAS_STEP enabled=1\nKVLAB_PPAS_PRESSURE active_prefills=2 decodes=1 budget=2048'
        self.assertTrue(assess(log, 'ppas')['comparison_ready'])
        self.assertFalse(assess(log+'\nKVLAB_PPAS_INIT enabled=0', 'ppas')['activation_verified'])
        self.assertFalse(assess(log, 'fixed_2k')['activation_verified'])

    def test_observer_rejects_unknown_source(self):
        with self.assertRaises(ValueError):
            instrument('class Scheduler: pass')

    def test_real_scheduler_observation_tracks_pressure_once(self):
        # Exercise the exact observer insertion and pressure expression using
        # a minimal scheduler fixture, without importing GPU dependencies.
        import contextlib
        import io
        source = '''import os
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
            num_prefill_running = sum(r.prefill for r in self.running)
            num_running_decodes = len(self.running) - num_prefill_running
            num_active_prefills = num_prefill_running + len(self.waiting)
            under_prefill_pressure = num_active_prefills >= 2 and num_running_decodes > 0
            ppas_prefill_budget = 2048 if under_prefill_pressure else self.max_num_scheduled_tokens
            # Tracks aggregate prefill tokens admitted in this scheduler step.
            return ppas_prefill_budget
'''
        from unittest.mock import patch
        from types import SimpleNamespace
        namespace = {}
        exec(compile(instrument(source), '<scheduler-fixture>', 'exec'), namespace)
        output = io.StringIO()
        with patch.dict('os.environ', {'PPAS_ENABLED': '1'}), contextlib.redirect_stdout(output):
            scheduler = namespace['Scheduler']()
            self.assertEqual(scheduler.schedule(), 16384)
            scheduler.running = [SimpleNamespace(prefill=False)]
            scheduler.waiting = [object(), object()]
            self.assertEqual(scheduler.schedule(), 2048)
            self.assertEqual(scheduler.schedule(), 2048)
        log = output.getvalue()
        self.assertEqual(log.count('KVLAB_PPAS_STEP'), 1)
        self.assertEqual(log.count('KVLAB_PPAS_PRESSURE'), 1)
        self.assertTrue(assess(log, 'ppas')['comparison_ready'])
