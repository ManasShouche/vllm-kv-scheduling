"""Retrospective SLO surfaces and interval-censored pressure diagnostics.

Never replays a new scheduler or infers causal effects from counter overlap.
Historical experiment gates remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path

from .analyze import extract_metrics, request_stats
from .audit import load_artifacts
from .compare import load_matrix


TTFT_GRID = (.5, 1., 1.5, 2.5, 5., 30., 90.)
GAP_GRID = (.05, .1, .2, .5, 1., 5.)


def preemption_intervals(metrics):
    """An increment occurred somewhere between scrapes, not at the latter tick."""
    intervals, previous = [], None
    missing = resets = 0
    for sample in metrics:
        value = extract_metrics(sample.get('raw_prometheus', '')).get('preemptions')
        if value is None:
            missing += 1
            continue
        if previous is not None:
            tick, counter = previous
            if sample['t_s'] <= tick:
                raise ValueError('Counter scrape times must strictly increase')
            delta = value - counter
            if delta < 0:
                resets += 1
            elif delta > 0:
                intervals.append({'start_s': tick, 'end_s': sample['t_s'], 'delta': delta})
        previous = (sample['t_s'], value)
    return intervals, {'missing_counter_samples': missing, 'counter_resets': resets}


def phase_analysis(rows, phase, intervals, ttft_grid=TTFT_GRID, gap_grid=GAP_GRID):
    cohort = [r for r in rows if r['phase'] == phase and not r['warmup']]
    stats = [(r, request_stats(r)) for r in cohort]
    surface = []
    for ttft in ttft_grid:
        for gap in gap_grid:
            # Arrival-relative latency includes client dispatch delay. Failed,
            # rejected, censored, and unobservable streams cannot pass.
            passed = sum(r['status'] == 'ok' and s['ttft_arrival_s'] is not None
                         and s['ttft_arrival_s'] <= ttft
                         and s['max_event_gap_s'] is not None
                         and s['max_event_gap_s'] <= gap for r, s in stats)
            surface.append({'ttft_s': ttft, 'gap_s': gap, 'passes': passed,
                            'arrivals': len(cohort),
                            'fraction': passed / len(cohort) if cohort else None})
    # All observed gaps, including failed requests; missing/censored tails are
    # not imputed. Fix 100ms for this diagnostic rather than picking a best value.
    stalled = overlap = 0
    for row in cohort:
        ticks = row.get('event_times_s', [])
        gaps = [(a, b) for a, b in zip(ticks, ticks[1:]) if b-a > .1]
        stalled += bool(gaps)
        overlap += any(a < i['end_s'] and b > i['start_s']
                       for a, b in gaps for i in intervals)
    return {'phase': phase, 'arrivals': len(cohort), 'surface': surface,
            'requests_with_observed_gap_over_100ms': stalled,
            'requests_with_gap_overlapping_preemption_interval': overlap,
            'unsuccessful': sum(r['status'] != 'ok' for r in cohort)}


def batch_analysis(root, workload):
    # Requires complete matched matrix and verifies source schedules, not just
    # saved summaries. Different batches remain distinct experiment strata.
    load_matrix(root, workload=workload)
    arms = []
    for data in sorted(Path(root).glob('*/data')):
        meta, rows, metrics = load_artifacts(data)
        intervals, coverage = preemption_intervals(metrics)
        hashes = {name: hashlib.sha256((data/name).read_bytes()).hexdigest()
                  for name in ('run.json', 'requests.jsonl', 'metrics.jsonl')}
        arms.append({'labels': meta['labels'], 'source_sha256': hashes,
                     'execution_kind': meta['provenance'].get('execution_kind'),
                     'preemption_intervals': intervals, 'counter_coverage': coverage,
                     'phases': [phase_analysis(rows, phase['name'], intervals)
                                for phase in meta['config']['phases']]})
    return {'batch': Path(root).name, 'arms': arms}


def write_report(batches, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    result = {'analysis_kind': 'retrospective_existing_runs_not_new_policy_evaluation',
              'latency_origin': 'scheduled_arrival',
              'semantics': 'SSE text-event gaps, not token ITL',
              'causal_claim': False, 'confidence_intervals': None,
              'grid_selected_after_historical_results': True, 'batches': batches}
    (destination/'surfaces.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = ['# Retrospective research diagnostics', '',
             'Existing GPU observations only. No new policy was simulated or executed.',
             'The threshold grid is exploratory, not a replacement for the original frozen gates.',
             'Arrival-relative TTFT; failures stay in denominators; no pooled seed inference.', '',
             '| Batch | Policy | Seed | Phase | SLO passes at 2.5s / 100ms | Requests with >100ms gap | Of those, overlapping preemption interval |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for batch in batches:
        for arm in batch['arms']:
            for phase in arm['phases']:
                point = next(p for p in phase['surface'] if p['ttft_s'] == 2.5 and p['gap_s'] == .1)
                lines.append(f"| {batch['batch']} | {arm['labels']['policy']} | {arm['labels']['seed']} | {phase['phase']} | {point['passes']}/{point['arrivals']} | {phase['requests_with_observed_gap_over_100ms']} | {phase['requests_with_gap_overlapping_preemption_interval']} |")
    lines += ['', 'Counter increments only localize preemption to a scrape interval. An overlap',
              'neither identifies the evicted request nor proves the cause of a stall.',
              'Missing scrapes widen intervals; resets and coverage are retained in the JSON.',
              'No overlap does not exclude delayed recomputation effects. Server iteration',
              'and per-request recomputation telemetry is required before a causal claim.', '']
    (destination/'diagnostics.md').write_text('\n'.join(lines))


def plot_surfaces(batches, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    pairs = []
    for batch in batches:
        groups = {}
        for arm in batch['arms']:
            label = arm['labels']
            groups.setdefault((label['seed'], label['capacity']), {})[label['policy']] = arm
        for (seed, capacity), arms in groups.items():
            if {'fixed_2k', 'fixed_512'} <= arms.keys():
                a, b = [next(p for p in arms[k]['phases'] if p['phase'] == 'long_output')
                        for k in ('fixed_2k', 'fixed_512')]
                if not a['arrivals'] or not b['arrivals']:
                    continue
                grid = np.array([100*(q['fraction']-p['fraction'])
                                 for p, q in zip(a['surface'], b['surface'])])
                pairs.append((seed, capacity, grid.reshape(len(TTFT_GRID), len(GAP_GRID))))
    if not pairs:
        return
    fig, axes = plt.subplots(1, len(pairs), figsize=(6*len(pairs), 5), squeeze=False,
                             constrained_layout=True)
    for ax, (seed, capacity, grid) in zip(axes[0], pairs):
        im = ax.imshow(grid, vmin=-100, vmax=100, cmap='RdBu', aspect='auto')
        ax.set_xticks(range(len(GAP_GRID)), GAP_GRID)
        ax.set_yticks(range(len(TTFT_GRID)), TTFT_GRID)
        ax.set_xlabel('Maximum SSE event gap allowed (s)')
        ax.set_ylabel('Arrival-to-first-response allowed (s)')
        ax.set_title(f'Seed {seed}, {capacity}, long outputs')
        for i in range(len(TTFT_GRID)):
            for j in range(len(GAP_GRID)):
                ax.text(j, i, f'{grid[i,j]:+.0f}', ha='center', va='center',
                        color='white' if abs(grid[i,j]) > 65 else 'black', fontsize=9)
    fig.colorbar(im, ax=list(axes[0]), label='Fixed-512 minus fixed-2k SLO attainment (percentage points)')
    fig.suptitle('Historical sensitivity analysis — exploratory thresholds; no new GPU runs')
    for extension in ('png', 'pdf'):
        fig.savefig(Path(destination)/f'slo-surface.{extension}', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', nargs=2, action='append', required=True,
                        metavar=('RESULT_ROOT', 'FROZEN_WORKLOAD'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    batches = [batch_analysis(root, workload) for root, workload in args.batch]
    write_report(batches, args.output)
    if args.plots:
        plot_surfaces(batches, args.output)
    print(Path(args.output)/'diagnostics.md')


if __name__ == '__main__':
    main()
