"""Audit scheduler-to-client identity and recomputation observations; no causality claim."""
import argparse
import json
import re
from pathlib import Path

from .audit import load_artifacts, verify_records


def inspect_trace(arm, workload, allow_censored=False):
    arm = Path(arm)
    meta, rows, metrics = load_artifacts(arm/'data')
    verify_records(meta, rows, metrics, workload)
    if not meta['provenance']['server'].get('trace_enabled'):
        raise ValueError('Arm was not instrumented')
    if meta['provenance']['server']['async_scheduling'] != 'explicitly disabled for telemetry and matched controls':
        raise ValueError('Trace attribution requires synchronous scheduling')
    paths = list((arm/'trace').glob('trace-*.jsonl'))
    if len(paths) != 1:
        raise ValueError('Exactly one engine trace is required')
    client = {row['request_id']: row for row in rows}
    censored = {rid for rid, row in client.items() if allow_censored and row['status'] in ('timeout', 'censored')}
    successful = set(client)-censored
    stats = {rid: dict(emitted_tokens=0, recompute_tokens=0, preemptions=0,
                      result_gaps_over_100ms=0, gaps_with_own_preemption=0,
                      max_result_gap_s=None) for rid in client}
    ids, last_output, last_preempt, pending = {}, {}, {}, {}
    seen_preemptions = 0
    seq = 0
    previous_time = -1
    origin = meta['origin_perf_counter_ns']
    first_seen = set()
    ended = False

    def identify(internal):
        match = re.fullmatch(r'cmpl-(s\d+-\d{6})-0-[0-9a-f]{8}', internal)
        if not match or match[1] not in client:
            raise ValueError('Unmapped engine request ID: '+internal)
        rid = match[1]
        response = client[rid].get('server_response_id')
        if response != 'cmpl-'+rid and not (rid in censored and response is None):
            raise ValueError('Client response ID does not match engine identity')
        if rid in ids and ids[rid] != internal:
            raise ValueError('Multiple engine requests map to one client request')
        ids[rid] = internal
        return rid

    with paths[0].open() as stream:
        for line in stream:
            event = json.loads(line)
            seq += 1
            if ended or type(event['time_ns']) is not int or event['sequence'] != seq or event['time_ns'] < previous_time:
                raise ValueError('Missing/reordered telemetry events')
            previous_time = event['time_ns']
            tick = (event['time_ns']-origin)/1e9
            kind = event['event']
            if seq == 1 and (kind != 'header' or event.get('schema') != 1):
                raise ValueError('Trace header absent or unsupported')
            if kind == 'header' and (seq != 1 or event.get('clock') != 'perf_counter_ns_same_linux_host'):
                raise ValueError('Unsupported clock or duplicate header')
            if kind == 'footer':
                ended = True
                if event.get('pending_steps') != 0:
                    raise ValueError('Trace ended with outstanding steps')
            if kind == 'scheduled':
                if event['step'] in pending:
                    raise ValueError('Duplicate step')
                pending[event['step']] = event['requests']
                for row in event['requests']:
                    rid = identify(row['request_id'])
                    if not 0 <= row['recompute_scheduled_tokens'] <= row['scheduled_tokens']:
                        raise ValueError('Invalid recomputation count')
            elif kind == 'completed':
                scheduled = pending.pop(event['step'])
                scheduled_ids = {row['request_id'] for row in scheduled}
                for row in scheduled:
                    stats[identify(row['request_id'])]['recompute_tokens'] += row['recompute_scheduled_tokens']
                for row in event['outputs']:
                    rid = identify(row['request_id'])
                    if row['request_id'] not in scheduled_ids or row['new_tokens'] not in (0, 1):
                        raise ValueError('Unexpected output or multi-token step')
                    if row['new_tokens']:
                        if tick < client[rid]['arrival_s'] or (rid not in censored and tick > client[rid]['finished_s']+.05):
                            raise ValueError('Server/client clocks or request lifetimes disagree')
                        if rid in last_output:
                            gap = tick-last_output[rid]
                            stats[rid]['max_result_gap_s'] = max(stats[rid]['max_result_gap_s'] or 0, gap)
                            if gap > .1:
                                stats[rid]['result_gaps_over_100ms'] += 1
                                stats[rid]['gaps_with_own_preemption'] += last_preempt.get(rid, -1) > last_output[rid]
                        last_output[rid] = tick
                        stats[rid]['emitted_tokens'] += row['new_tokens']
                    if row['finished']:
                        first_seen.add(rid)
            elif kind == 'preempt':
                rid = identify(event['request_id'])
                stats[rid]['preemptions'] += 1
                seen_preemptions += 1
                last_preempt[rid] = tick
            elif kind not in ('header', 'footer'):
                raise ValueError('Unknown trace event')
    if seq == 0 or pending or not successful <= set(ids) or not successful <= first_seen:
        raise ValueError('Incomplete trace/request lifecycle')
    for rid, row in client.items():
        if rid in censored:
            continue  # Server may finish after client cancellation; usage is unavailable.
        if row['status'] != 'ok' or stats[rid]['emitted_tokens'] != row['output_tokens']:
            raise ValueError('Trace token totals disagree with successful client usage')
    return {'ready': True, 'requests': stats, 'trace_events': seq,
            'preemptions': seen_preemptions, 'all_requests_mapped': set(ids) == set(client),
            'complete_client_lifecycles': not censored,
            'censored_requests': sorted(censored),
            'censoring_note': 'Timeout/drain-cancelled requests remain failures. Observed engine events only; no full lifecycle or client-token-total claim for them.',
            'timing_semantics': 'CPU result availability on same host, not CUDA time or SSE emission',
            'causal_claim': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm')
    parser.add_argument('--workload', required=True)
    parser.add_argument('--allow-censored', action='store_true')
    args = parser.parse_args()
    result = inspect_trace(args.arm, args.workload, args.allow_censored)
    (Path(args.arm)/'trace-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'requests'}))


if __name__ == '__main__':
    main()
