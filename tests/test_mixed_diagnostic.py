import contextlib
import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace as O
from lab.eviction import Eviction
from lab.workload import arrivals,validate

class DiagnosticTests(unittest.TestCase):
    def test_real_controller_changes_victim_and_records_excluded_prefix(self):
        rows=[O(request_id=str(i),num_computed_tokens=n,num_preemptions=0) for i,n in enumerate([16,2048,8192])]
        s=O(running=rows,kv_cache_manager=O(get_blocks=lambda rid:O(blocks=([1]*8,))))
        log=io.StringIO()
        with contextlib.redirect_stdout(log):
            h=Eviction('evict_cost');self.assertEqual(h.choose(s,1),1)
        x=json.loads(next(line.split(' ',1)[1] for line in log.getvalue().splitlines() if line.startswith('KVLAB_EVICTION_DECISION ')))
        self.assertFalse(x['requests'][0]['eligible'])
        self.assertEqual(x['selected'],1)
        self.assertEqual(x['default'],2)
        self.assertIn('event=changed',log.getvalue())

    def test_homogeneous_tail_and_singleton_have_no_alternative(self):
        for tokens,start in [([8192,8192],0),([2048,8192],1)]:
            rows=[O(request_id=str(i),num_computed_tokens=n,num_preemptions=0) for i,n in enumerate(tokens)]
            s=O(running=rows,kv_cache_manager=O(get_blocks=lambda rid:O(blocks=([1],))))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(Eviction('evict_cost').choose(s,start),len(rows)-1)

    def test_mixture_is_reproducible_without_changing_arrivals(self):
        config=json.loads(Path('configs/eviction-mixed-diagnostic.json').read_text());validate(config)
        mixed=arrivals(config,102);self.assertEqual(mixed,arrivals(config,102))
        self.assertEqual({r['prompt_tokens'] for r in mixed},{2048,8192,12288})
        for phase in config['phases']:del phase['prompt_tokens_choices']
        fixed=arrivals(config,102)
        self.assertEqual([r['arrival_s'] for r in mixed],[r['arrival_s'] for r in fixed])
        for phase in config['phases']:phase['prompt_tokens_choices']=[True]
        with self.assertRaises(ValueError):validate(config)
        for phase in config['phases']:phase['prompt_tokens_choices']=[16000]
        with self.assertRaises(ValueError):validate(config)
