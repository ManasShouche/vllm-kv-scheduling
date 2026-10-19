import unittest

from lab.research import phase_analysis, preemption_intervals
from lab.feasibility import Option, State, choose


class ResearchTests(unittest.TestCase):
    def test_arrival_latency_and_failures_cannot_be_hidden(self):
        rows = [dict(phase='p', warmup=False, status=status, arrival_s=0,
                     actual_dispatch_s=1., event_times_s=[1.1, 1.15])
                for status in ('ok', 'timeout')]
        result = phase_analysis(rows, 'p', [], ttft_grid=(.5, 2.), gap_grid=(.1,))
        self.assertEqual([p['passes'] for p in result['surface']], [0, 1])
        self.assertEqual([p['arrivals'] for p in result['surface']], [2, 2])

    def test_scrape_uncertainty_missing_samples_and_counter_reset(self):
        samples = [{'t_s': t, 'raw_prometheus': f'vllm:num_preemptions_total {c}\n'}
                   for t, c in [(0, 2), (3, 4), (4, 0), (5, 1)]]
        samples.insert(1, {'t_s': 1, 'error': 'missing'})
        intervals, coverage = preemption_intervals(samples)
        self.assertEqual(intervals, [dict(start_s=0, end_s=3, delta=2),
                                     dict(start_s=4, end_s=5, delta=1)])
        self.assertEqual(coverage, dict(missing_counter_samples=1, counter_resets=1))
        rows = [dict(phase='p', warmup=False, status='ok', arrival_s=0,
                     actual_dispatch_s=0, event_times_s=[1., 1.2, 1.5])]
        result = phase_analysis(rows, 'p', intervals)
        self.assertEqual(result['requests_with_gap_overlapping_preemption_interval'], 1)

    def test_unobservable_gap_never_passes(self):
        row = dict(phase='p', warmup=False, status='ok', arrival_s=0,
                   actual_dispatch_s=0, event_times_s=[.1])
        self.assertTrue(all(p['passes'] == 0 for p in phase_analysis([row], 'p', [])['surface']))


class FeasibilityTests(unittest.TestCase):
    def setUp(self):
        self.options = [Option(0, 10, 200, 0), Option(256, 30, 120, 16),
                        Option(512, 60, 90, 32), Option(1024, 100, 100, 64)]

    def test_respects_joint_constraints_and_decode_growth(self):
        state = State(70, 150, 60, 20, True)
        self.assertEqual(choose(self.options, state)['prefill_tokens'], 512)
        constrained = State(70, 150, 40, 20, True)
        self.assertEqual(choose(self.options, constrained)['prefill_tokens'], 256)

    def test_conflict_and_overload_are_explicit(self):
        result = choose(self.options, State(40, 100, 100, 0, True))
        self.assertEqual(result['action'], 'defer_prefill')
        self.assertFalse(result['joint_model_feasible'])
        self.assertEqual(choose(self.options, State(5, 100, 100, 0, True))['action'], 'overload')
        self.assertEqual(choose(self.options, State(None, 100, 1, 2, True))['action'], 'overload')

    def test_nonmonotone_cost_and_no_active_decode(self):
        options = [Option(0, 5, 200, 0), Option(256, 80, 120, 16), Option(512, 40, 90, 32)]
        self.assertEqual(choose(options, State(50, 100, 100, 0, True))['prefill_tokens'], 512)
        self.assertEqual(choose(self.options, State(None, 150, 100, 0, True))['prefill_tokens'], 1024)

    def test_uncalibrated_and_invalid_inputs_fail_closed(self):
        self.assertEqual(choose(self.options, State(70, 150, 100, 0, False))['action'], 'requires_calibration')
        for value in (float('nan'), float('inf'), -1, True):
            with self.assertRaises(ValueError):
                Option(1, value, 200, 1)
        with self.assertRaises(ValueError):
            choose(self.options + [self.options[0]], State(70, 150, 100, 0, True))

    def test_all_selected_actions_satisfy_constraints_over_state_grid(self):
        # Independently check the feasible set including boundary equality.
        for decode in (0, 10, 30, 60, 100, None):
            for deadline in (-1, 90, 100, 120, 200):
                for free in (0, 16, 32, 64, 100):
                    state = State(decode, deadline, free, 16, True)
                    result = choose(self.options, state)
                    allowed = [o.prefill_tokens for o in self.options if o.prefill_tokens
                               and (decode is None or o.step_upper_ms <= decode)
                               and o.first_response_upper_ms <= deadline
                               and o.additional_kv_blocks + 16 <= free]
                    self.assertEqual(result['joint_model_feasible'], bool(allowed))
                    if allowed:
                        self.assertEqual(result['prefill_tokens'], max(allowed))
