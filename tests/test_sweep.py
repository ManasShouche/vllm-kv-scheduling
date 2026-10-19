"""Check cleanup before reporting and bounded reporting failures without a GPU."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from lab.sweep import main, select_arms


class SweepTests(unittest.TestCase):
    def run_sweep(self, failure=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workload = root/'workload'
            workload.mkdir()
            (workload/'config.json').write_text(json.dumps({
                'seeds': [41], 'policies': {'fixed_2k': 2048},
                'kv_capacities_bytes': {'roomy': 1024}}))
            commands = []
            def execute(command, **kwargs):
                commands.append((command, kwargs))
                if command[:2] == ['docker', 'logs']:
                    kwargs['stdout'].write('KVLAB_PPAS_INIT enabled=0\n')
                if failure and failure in command:
                    raise subprocess.TimeoutExpired(command, kwargs['timeout'])
            telemetry = Mock()
            telemetry.poll.return_value = None
            argv = ['lab.sweep', '--workload', str(workload), '--output', str(root/'out'), '--one-arm']
            with patch('sys.argv', argv), patch('lab.sweep.identity', return_value={}), \
                    patch('lab.sweep.signal.signal'), patch('lab.sweep.healthy', side_effect=[False, True]), \
                    patch('lab.sweep.subprocess.run', side_effect=execute), \
                    patch('lab.sweep.subprocess.Popen', return_value=telemetry):
                if failure:
                    with self.assertRaises(subprocess.TimeoutExpired):
                        main()
                else:
                    main()
            arm = next((root/'out').glob('00-*'))
            status = json.loads((arm/'status.json').read_text())
            self.assertTrue((arm/'server.log').exists())
            stop = next(i for i,(c,k) in enumerate(commands) if c[:2] == ['docker','stop'])
            reports = [(i,c,k) for i,(c,k) in enumerate(commands)
                       if 'lab.analyze' in c or 'lab.audit' in c]
            self.assertTrue(reports)
            for i,c,k in reports:
                self.assertGreater(i, stop)
                self.assertGreater(k['timeout'], 0)
                self.assertLessEqual(k['timeout'], 120)
            return status

    def test_report_timeout_preserves_logs_and_failed_status_after_gpu_cleanup(self):
        status = self.run_sweep('lab.analyze')
        self.assertEqual(status['state'], 'failed')
        self.assertIn('TimeoutExpired', status['error'])

    def test_success_requires_report_and_audit_after_gpu_cleanup(self):
        self.assertEqual(self.run_sweep()['state'], 'complete')


class SelectionTests(unittest.TestCase):
    config = {"seeds": [41, 42, 43],
              "policies": {"fixed_2k": 2048, "fixed_16k": 16384, "ppas": 16384},
              "kv_capacities_bytes": {"roomy": 8589934592, "constrained": 1073741824}}

    def test_exact_constrained_ppas_arm(self):
        self.assertEqual(select_arms(self.config, policy="ppas", capacity="constrained", seed=42),
                         [{"policy": "ppas", "capacity": "constrained", "seed": 42}])

    def test_full_matrix_is_complete_and_repeatable(self):
        arms = select_arms(self.config)
        self.assertEqual(len(arms), 18)
        self.assertEqual(arms, select_arms(self.config))
        self.assertEqual(len({tuple(sorted(a.items())) for a in arms}), 18)

    def test_invalid_filters_and_conflicts_fail(self):
        for kwargs in ({"policy": "typo"}, {"capacity": "tiny"}, {"seed": 99},
                       {"one_arm": True, "policy": "ppas"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                select_arms(self.config, **kwargs)

    def test_dry_run_never_accesses_aws_or_creates_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"config.json").write_text(json.dumps(self.config))
            argv = ["lab.sweep", "--workload", tmp, "--output", str(root/"out"),
                    "--policy", "ppas", "--capacity", "constrained", "--seed", "41", "--dry-run"]
            with patch("sys.argv", argv), patch("lab.sweep.identity") as aws, \
                    patch("lab.sweep.signal.signal"), patch("builtins.print"):
                main()
            aws.assert_not_called()
            self.assertFalse((root/"out").exists())

class ComparisonScreenTests(unittest.TestCase):
    def test_valid_failed_outcome_does_not_suppress_second_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'config.json').write_text(json.dumps({'seeds':[101],
                'policies':{'fixed_512':512,'cap_2':512},'kv_capacities_bytes':{'constrained':1024}}))
            commands=[]
            def execute(command,**kwargs):
                commands.append(command)
                if command[:2]==['docker','logs']:kwargs['stdout'].write('KVLAB_PPAS_INIT enabled=0\n')
                if 'lab.audit' in command:
                    self.assertNotIn('--pilot',command)
                    # A valid timeout is an unsuccessful policy outcome; the
                    # integrity-only audit returns success and retains that count.
                    directory=Path(command[3]);directory.mkdir(parents=True,exist_ok=True)
                    (directory/'audit.json').write_text(json.dumps({'artifact_valid':True,'unsuccessful_requests':1}))
            telemetry=Mock();telemetry.poll.return_value=None
            with patch('sys.argv',['sweep','--workload',tmp,'--output',tmp+'/out','--comparison-screen','--trace-all']),patch('lab.sweep.identity',return_value={}),patch('lab.sweep.signal.signal'),patch('lab.sweep.healthy',side_effect=[False,True,False,True]),patch('lab.sweep.subprocess.run',side_effect=execute),patch('lab.sweep.subprocess.Popen',return_value=telemetry):
                main()
            self.assertEqual(sum('lab.collect' in c for c in commands),2)
            traces=[c for c in commands if 'lab.trace_audit' in c]
            self.assertEqual(len(traces),2)
            self.assertTrue(all('--allow-censored' in c for c in traces))
            self.assertTrue(all(json.loads(p.read_text())['state']=='complete' for p in (root/'out').glob('*/status.json')))
