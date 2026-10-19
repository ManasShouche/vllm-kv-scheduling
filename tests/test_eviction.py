import unittest
from types import SimpleNamespace as O
from pathlib import Path
from lab.eviction import victim_index
from lab.activation import assess


def req(tokens, repeats=0):
    return O(num_computed_tokens=tokens,num_preemptions=repeats)


class EvictionTests(unittest.TestCase):
    def test_scheduled_prefix_never_evicted(self):
        rows=[req(1),req(2000),req(4000)]
        self.assertEqual(victim_index(rows,1,lambda r:1),1)

    def test_ties_and_small_differences_keep_upstream_victim(self):
        self.assertEqual(victim_index([req(100),req(104)],0,lambda r:1),1)
        self.assertEqual(victim_index([req(100),req(100)],0,lambda r:1),1)

    def test_repeat_penalty_changes_cost_ablation_choice(self):
        rows=[req(100,4),req(300)]
        self.assertEqual(victim_index(rows,0,lambda r:1),0)
        self.assertEqual(victim_index(rows,0,lambda r:1,True),1)

    def test_zero_block_victim_is_excluded_unless_all_empty(self):
        rows=[req(0),req(1000)]
        self.assertEqual(victim_index(rows,0,lambda r:int(r.num_computed_tokens>0)),1)
        self.assertEqual(victim_index(rows,0,lambda r:0),1)

    def test_cursor_and_removal_invariant_exhaustive(self):
        for n in range(1,9):
            for start in range(n):
                rows=[req((n-i)*100,i%3) for i in range(n)]
                for fair in (False,True):
                    index=victim_index(rows,start,lambda r:1,fair)
                    self.assertGreaterEqual(index,start)
                    kept=rows.copy();kept.pop(index)
                    self.assertEqual(kept[:start],rows[:start])
        with self.assertRaises(ValueError):victim_index([],0,lambda r:1)

    def test_activation_requires_actual_changed_victim_for_comparison(self):
        log='KVLAB_PPAS_INIT enabled=0\nKVLAB_EVICTION event=init mode=evict_fair\n'
        self.assertTrue(assess(log,'evict_fair')['activation_verified'])
        self.assertFalse(assess(log,'evict_fair')['comparison_ready'])
        log+='KVLAB_EVICTION event=changed mode=evict_fair\n'
        self.assertTrue(assess(log,'evict_fair')['comparison_ready'])
        self.assertFalse(assess(log,'fixed_512')['activation_verified'])

    def test_entire_pinned_patch_stack(self):
        from scripts.extend_scheduler import extend as a
        from scripts.instrument_scheduler import instrument as b
        from scripts.instrument_recompute import instrument as c
        from scripts.extend_headroom import extend as d
        from scripts.extend_eviction import extend as e
        p=Path('evidence/upstream-check/vllm/v1/core/sched/scheduler.py')
        if not p.exists():self.skipTest('Pinned source optional locally')
        patched=e(d(c(b(a(p.read_text())))))
        self.assertIn('self.running.pop(victim_index)',patched)
        with self.assertRaises(ValueError):e(patched)
        with self.assertRaises(ValueError):e('class Scheduler: pass')
