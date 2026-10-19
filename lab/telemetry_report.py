"""One-seed ABBA observer-overhead screen; not independent efficacy trials."""
import argparse
import json
from pathlib import Path
from statistics import mean

from .analyze import summarize
from .audit import load_artifacts, verify_records
from .trace_audit import inspect_trace


def report(root, workload):
    arms = sorted(Path(root).glob('[0-9]*'))
    if len(arms) != 4:
        raise ValueError('ABBA requires all four arms; retain partial results separately')
    reference = None
    results = []
    expected = (False, True, True, False)
    for index, arm in enumerate(arms):
        status = json.loads((arm/'status.json').read_text())
        if status['state'] != 'complete' or status['arm'].get('trace_enabled') != expected[index]:
            raise ValueError('Incomplete arm or wrong instrumentation order')
        meta, rows, metrics = load_artifacts(arm/'data')
        verify_records(meta, rows, metrics, workload)
        server = meta['provenance']['server']
        identity = [meta['config'], meta['labels'], meta['provenance']['workload'],
                    meta['provenance']['schedule_sha256'], server['image_id'], server['instance_id'],
                    server['model_revision'], server['async_scheduling']]
        if reference is None:
            reference = identity
        if identity != reference or server.get('trace_enabled') != expected[index]:
            raise ValueError('Unmatched workload, host, image, or instrumentation')
        if any(row['status'] != 'ok' for row in rows):
            raise ValueError('Unsuccessful requests prevent observer-overhead screening')
        if not expected[index] and list((arm/'trace').glob('*.jsonl')):
            raise ValueError('Disabled control unexpectedly emitted a trace')
        trace = inspect_trace(arm, workload) if expected[index] else None
        results.append({'index': index, 'trace_enabled': expected[index],
                        'summary': summarize(meta, rows, metrics), 'trace': trace})
    off = [r['summary'] for r in results if not r['trace_enabled']]
    on = [r['summary'] for r in results if r['trace_enabled']]
    throughput = mean(s['completed_output_tokens_per_drain_inclusive_s'] for s in on)/mean(s['completed_output_tokens_per_drain_inclusive_s'] for s in off)
    phases = {}
    for phase in off[0]['phases']:
        phases[phase] = {
            'ttft_p95_ratio_on_over_off': mean(s['phases'][phase]['ttft_p95_s'] for s in on)/mean(s['phases'][phase]['ttft_p95_s'] for s in off),
            'median_worst_gap_ratio_on_over_off': mean(s['phases'][phase]['request_max_event_gap_p50_s'] for s in on)/mean(s['phases'][phase]['request_max_event_gap_p50_s'] for s in off)}
    passed = throughput >= .95 and all(p['ttft_p95_ratio_on_over_off'] <= 1.1 and p['median_worst_gap_ratio_on_over_off'] <= 1.1 for p in phases.values())
    result = {'design': 'same-seed off/on/on/off; descriptive overhead screen only',
              'throughput_ratio_on_over_off': throughput, 'phase_ratios': phases,
              'passes_overhead_screen': passed,
              'preemption_observed_in_both_traced_arms': all(r['trace']['preemptions'] > 0 for r in results if r['trace_enabled']),
              'causal_claim': False, 'arms': results}
    (Path(root)/'telemetry-comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--workload', required=True)
    args = parser.parse_args()
    result = report(args.root, args.workload)
    print(json.dumps({k: v for k, v in result.items() if k != 'arms'}, indent=2))


if __name__ == '__main__':
    main()
