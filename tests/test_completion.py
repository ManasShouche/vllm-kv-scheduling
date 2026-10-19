import json
import os
from pathlib import Path
import subprocess
import unittest
from lab.sweep import select_arms
from lab.workload import validate


class CompletionTests(unittest.TestCase):
    def test_validation_filters_rejected_before_host_access(self):
        for name,value in [('PILOT_SEED','43'),('PILOT_CAPACITY','constrained'),('PILOT_POLICY','fixed_512')]:
            env={**os.environ,'PILOT_MODE':'validation',name:value}
            result=subprocess.run(['bash','scripts/run_first_arm_on_ec2.sh'],env=env,text=True,capture_output=True,timeout=5)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('filters are not allowed',result.stderr)

    def test_full_validation_uses_only_new_seeds_and_matched_controls(self):
        config=json.loads(Path('configs/final-validation.json').read_text())
        validate(config)
        arms=select_arms(config)
        expected={(policy,capacity,seed) for policy in ['fixed_2k','fixed_512']
                  for capacity in ['roomy','constrained'] for seed in [44,45,46]}
        self.assertEqual({(a['policy'],a['capacity'],a['seed']) for a in arms},expected)
        self.assertEqual(len(arms),len(expected))
