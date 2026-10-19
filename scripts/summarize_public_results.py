"""Recompute the public result table from derived data, without cloud access."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'results' / 'heldout'
gates = json.loads((root / 'independent-gates.json').read_text())
fairness = json.loads((root / 'independent-fairness.json').read_text())
for seed, row in gates.items():
    reduction = 100 * (1 - row['candidate_recompute'] / row['baseline_recompute'])
    delay = fairness[seed]['cohorts']['all']['completion_delta_s']['max']
    print(f"Seed {seed}: {reduction:.2f}% less recomputation; "
          f"screen={'PASS' if row['passes'] else 'FAIL'}; "
          f"max paired completion increase={delay:.3f}s")
print('Full screen:', 'PASS' if all(v['passes'] for v in gates.values()) else 'FAIL')
print('Derived data only; raw trace reproduction requires the separately preserved archive.')
