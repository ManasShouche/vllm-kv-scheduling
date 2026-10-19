import ast
import unittest
from types import SimpleNamespace as Obj
from pathlib import Path
from lab.headroom import Headroom
from lab.activation import assess
from scripts.extend_headroom import extend


def request(name, tokens):
    return Obj(request_id=name, num_tokens=tokens)


def scheduler(running, free, allocated):
    return Obj(running=running, block_size=16, max_model_len=16384,
        kv_cache_manager=Obj(block_pool=Obj(get_num_free_blocks=lambda: free),
            get_blocks=lambda rid: Obj(blocks=([None]*allocated.get(rid, 0),))))


class HeadroomTests(unittest.TestCase):
    def test_reserves_unfinished_prompt_not_only_next_chunk(self):
        h = Headroom('reserve_static')
        s = scheduler([request('active', 8192)], 700, {'active': 32})
        # Existing prompt needs 496 blocks, incoming prompt needs 528.
        self.assertFalse(h.allows(s, request('new', 8192)))

    def test_exact_boundary_and_allocated_current_step(self):
        h = Headroom('reserve_static')
        s = scheduler([request('a', 8193)], 545, {'a': 512})
        self.assertTrue(h.allows(s, request('b', 8192)))
        s.kv_cache_manager.block_pool.get_num_free_blocks = lambda: 544
        self.assertFalse(h.allows(s, request('b', 8192)))

    def test_empty_running_does_not_deadlock_and_does_not_mutate_queue(self):
        h = Headroom('reserve_feedback')
        s = scheduler([], 0, {})
        self.assertTrue(h.allows(s, request('large', 16000)))
        self.assertEqual(s.running, [])

    def test_feedback_bounded_and_decay_requires_useful_tokens(self):
        h = Headroom('reserve_feedback')
        for _ in range(8):
            h.preempt()
        self.assertEqual(h.horizon, 2048)
        for _ in range(100):
            h.completed({})
        self.assertEqual(h.horizon, 2048)
        h.completed({0: Obj(outputs=[Obj(new_token_ids=[1]*4096)])})
        self.assertEqual(h.horizon, 1920)
        h.preempt()
        self.assertEqual(h.progress, 0)
        h.completed({0: Obj(outputs=[Obj(new_token_ids=[1]*100000)])})
        self.assertEqual(h.horizon, 256)

    def test_feedback_changes_admission_static_ablation_does_not(self):
        s = scheduler([request('a', 8192)], 560, {'a': 512})
        for mode, expected in [('reserve_static', True), ('reserve_feedback', False)]:
            h = Headroom(mode)
            self.assertTrue(h.allows(s, request('b', 8192)))
            h.preempt()
            self.assertEqual(h.allows(s, request('b', 8192)), expected)

    def test_activation_requires_runtime_and_intervention(self):
        base = 'KVLAB_PPAS_INIT enabled=0\n'
        init = base+'KVLAB_HEADROOM event=init mode=reserve_feedback\n'
        self.assertFalse(assess(init, 'reserve_feedback')['activation_verified'])
        step = init+'KVLAB_HEADROOM event=step mode=reserve_feedback\n'
        self.assertTrue(assess(step, 'reserve_feedback')['activation_verified'])
        self.assertFalse(assess(step, 'reserve_feedback')['comparison_ready'])
        defer = step+'KVLAB_HEADROOM event=defer mode=reserve_feedback\n'
        self.assertTrue(assess(defer, 'reserve_feedback')['comparison_ready'])
        self.assertFalse(assess(defer, 'fixed_512')['activation_verified'])
        self.assertFalse(assess(defer, 'reserve_static')['activation_verified'])

    def test_real_pinned_patch_stack_composes_and_rejects_drift(self):
        from scripts.extend_scheduler import extend as guard
        from scripts.instrument_scheduler import instrument as markers
        from scripts.instrument_recompute import instrument as trace
        source = Path('evidence/upstream-check/vllm/v1/core/sched/scheduler.py')
        if not source.exists():
            self.skipTest('Optional pinned source reference unavailable')
        patched = extend(trace(markers(guard(source.read_text()))))
        ast.parse(patched)
        self.assertIn('break  # Keep FCFS head', patched)
        with self.assertRaises(ValueError):
            extend(patched)
        with self.assertRaises(ValueError):
            extend('class Scheduler: pass')

    def test_trace_all_dry_run_preserves_three_matched_policies(self):
        import json
        import tempfile
        from unittest.mock import patch
        from lab.sweep import main
        from lab.workload import validate
        config = json.loads(Path('configs/headroom-screen.json').read_text())
        validate(config)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'config.json').write_text(json.dumps(config))
            with patch('sys.argv', ['sweep', '--workload', tmp, '--output', tmp+'/out', '--trace-all', '--dry-run']), patch('lab.sweep.identity') as aws, patch('builtins.print') as out:
                main()
            aws.assert_not_called()
            arms = json.loads(out.call_args[0][0])['arms']
            self.assertEqual({a['policy'] for a in arms}, set(config['policies']))
            self.assertTrue(all(a['trace_enabled'] for a in arms))

    def test_disabled_attach_does_not_import_vllm(self):
        from lab.headroom import attach
        self.assertIsNone(attach(None, ''))
