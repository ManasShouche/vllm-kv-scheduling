import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace as Obj
import unittest
from unittest.mock import patch

from lab.scheduler_trace import Trace
from lab.trace_audit import inspect_trace
from scripts.instrument_recompute import instrument
from lab.telemetry_report import report
from lab.sweep import main as sweep_main


class TraceTests(unittest.TestCase):
    def build(self, directory):
        directory = Path(directory)
        (directory/'trace').mkdir()
        origin = time.perf_counter_ns()-1_000_000_000
        trace = Trace(directory/'trace')
        rid = 'cmpl-s101-000001-0-abcdef12'
        req = Obj(request_id=rid, num_computed_tokens=0, num_prompt_tokens=8,
                  num_output_tokens=0, num_preemptions=0)
        scheduler = Obj(requests={rid: req}, running=[req], waiting=[],
                        kv_cache_manager=Obj(block_pool=Obj(get_num_free_blocks=lambda: 10)))
        first = Obj(num_scheduled_tokens={rid: 8})
        trace.scheduled(scheduler, first)
        trace.completed(first, {0: Obj(outputs=[Obj(request_id=rid, new_token_ids=[7], finish_reason=None)])})
        req.num_computed_tokens = 8
        req.num_output_tokens = 1
        trace.preempt(req)
        req.num_computed_tokens = 0
        req.num_preemptions = 1
        second = Obj(num_scheduled_tokens={rid: 8})
        trace.scheduled(scheduler, second)
        trace.completed(second, {})
        req.num_computed_tokens = 8
        third = Obj(num_scheduled_tokens={rid: 1})
        trace.scheduled(scheduler, third)
        trace.completed(third, {0: Obj(outputs=[Obj(request_id=rid, new_token_ids=[8], finish_reason=0)])})
        trace.close()
        rows = [dict(request_id='s101-000001', server_response_id='cmpl-s101-000001',
                     status='ok', output_tokens=2, arrival_s=0, finished_s=5)]
        meta = {'origin_perf_counter_ns': origin, 'provenance': {'server': {
            'trace_enabled': True, 'async_scheduling': 'explicitly disabled for telemetry and matched controls'}}}
        return trace.path, meta, rows

    def audit(self, root, meta, rows):
        with patch('lab.trace_audit.load_artifacts', return_value=(meta, rows, [])), \
                patch('lab.trace_audit.verify_records'):
            return inspect_trace(root, 'fixture')

    def test_preemption_recomputation_and_identity_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, meta, rows = self.build(tmp)
            result = self.audit(tmp, meta, rows)
            self.assertTrue(result['ready'])
            self.assertEqual(result['preemptions'], 1)
            self.assertEqual(result['requests']['s101-000001']['recompute_tokens'], 8)
            self.assertEqual(result['requests']['s101-000001']['emitted_tokens'], 2)
            self.assertNotIn('new_token_ids', path.read_text())
            self.assertNotIn('prompt_token_ids', path.read_text())

    def test_trace_loss_and_token_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, meta, rows = self.build(tmp)
            rows[0]['output_tokens'] = 3
            with self.assertRaisesRegex(ValueError, 'token totals'):
                self.audit(tmp, meta, rows)
            rows[0]['output_tokens'] = 2
            events = path.read_text().splitlines()
            path.write_text('\n'.join(events[:2]+events[3:])+'\n')
            with self.assertRaisesRegex(ValueError, 'Missing/reordered'):
                self.audit(tmp, meta, rows)

    def test_ambiguous_mapping_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, meta, rows = self.build(tmp)
            rows[0]['server_response_id'] = 'unrelated'
            with self.assertRaisesRegex(ValueError, 'Client response ID'):
                self.audit(tmp, meta, rows)

    def test_idle_steps_do_not_create_unmatched_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Trace(tmp)
            output = Obj(num_scheduled_tokens={})
            trace.scheduled(None, output)
            trace.completed(output, {})
            trace.close()
            self.assertEqual([json.loads(line)['event'] for line in trace.path.read_text().splitlines()], ['header', 'footer'])

    def test_missing_or_repeat_patch_fails_closed(self):
        with self.assertRaises(ValueError):
            instrument('class Scheduler: pass')
        with self.assertRaises(ValueError):
            instrument('_lab_trace = None')


class OverheadTests(unittest.TestCase):
    def test_abba_dry_run_never_accesses_aws(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'config.json').write_text(json.dumps({'seeds': [101],
                'policies': {'fixed_512': 512}, 'kv_capacities_bytes': {'constrained': 1024}}))
            with patch('sys.argv', ['sweep', '--workload', tmp, '--output', tmp+'/out', '--trace-overhead', '--dry-run']), \
                    patch('lab.sweep.identity') as aws, patch('lab.sweep.signal.signal'), patch('builtins.print') as printed:
                sweep_main()
            aws.assert_not_called()
            arms = json.loads(printed.call_args[0][0])['arms']
            self.assertEqual([a['trace_enabled'] for a in arms], [False, True, True, False])

    def test_overhead_failure_is_reported_without_relabeling(self):
        with tempfile.TemporaryDirectory() as tmp:
            metas = []
            for i, enabled in enumerate((False, True, True, False)):
                arm = Path(tmp)/f'{i:02d}-arm'
                arm.mkdir()
                (arm/'status.json').write_text(json.dumps({'state': 'complete', 'arm': {'trace_enabled': enabled}}))
                metas.append({'config': {}, 'labels': {}, 'provenance': {'workload': {},
                    'schedule_sha256': 'frozen', 'server': {'image_id': 'same', 'instance_id': 'same',
                    'model_revision': 'same', 'async_scheduling': 'disabled', 'trace_enabled': enabled}}})
            def summary(meta, rows, metrics):
                enabled = meta['provenance']['server']['trace_enabled']
                return {'completed_output_tokens_per_drain_inclusive_s': 90 if enabled else 100,
                        'phases': {'long': {'ttft_p95_s': 1, 'request_max_event_gap_p50_s': .1}}}
            with patch('lab.telemetry_report.load_artifacts', side_effect=[(m, [{'status': 'ok'}], []) for m in metas]), \
                    patch('lab.telemetry_report.verify_records'), patch('lab.telemetry_report.summarize', side_effect=summary), \
                    patch('lab.telemetry_report.inspect_trace', return_value={'preemptions': 1}):
                result = report(tmp, 'fixture')
            self.assertFalse(result['passes_overhead_screen'])
            self.assertEqual(result['throughput_ratio_on_over_off'], .9)
            self.assertFalse(result['causal_claim'])

class CensoredTraceTests(TraceTests):
    def test_timeout_is_explicit_and_successful_token_checks_remain_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, meta, rows = self.build(tmp)
            failed = dict(request_id='s101-000002', server_response_id=None,
                          status='timeout', output_tokens=None, arrival_s=0, finished_s=1)
            rows.append(failed)
            with patch('lab.trace_audit.load_artifacts', return_value=(meta, rows, [])), patch('lab.trace_audit.verify_records'):
                with self.assertRaises(ValueError):inspect_trace(tmp,'fixture')
                result=inspect_trace(tmp,'fixture',allow_censored=True)
                self.assertFalse(result['complete_client_lifecycles'])
                self.assertFalse(result['all_requests_mapped'])
                self.assertEqual(result['censored_requests'], ['s101-000002'])
                rows[0]['output_tokens']=99
                with self.assertRaisesRegex(ValueError,'token totals'):
                    inspect_trace(tmp,'fixture',allow_censored=True)

    def test_censoring_does_not_hide_unknown_ids_or_bad_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,meta,rows=self.build(tmp)
            rows[0]['status']='timeout'
            events=path.read_text().splitlines();path.write_text('\n'.join(events[:2]+events[3:])+'\n')
            with patch('lab.trace_audit.load_artifacts',return_value=(meta,rows,[])),patch('lab.trace_audit.verify_records'):
                with self.assertRaisesRegex(ValueError,'Missing/reordered'):
                    inspect_trace(tmp,'fixture',allow_censored=True)
