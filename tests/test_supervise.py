"""Mock host commands to check shutdown safeguards without touching AWS/systemd."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import subprocess

from lab.supervise import launch


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        (root/"lab").mkdir()
        self.host = {"region": "us-east-1", "instance_id": "i-fixture", "gpu": "fixture"}
        patches = [patch("lab.supervise.identity", return_value=self.host),
                   patch("lab.supervise.__file__", str(root/"lab/supervise.py")),
                   patch("lab.supervise.shutil.which", side_effect=lambda name: "/usr/bin/"+name),
                   patch("lab.supervise.subprocess.run"),
                   patch("lab.supervise.subprocess.check_output", side_effect=["stop\n", ""])]
        self.mocks = [p.start() for p in patches]
        for p in patches:
            self.addCleanup(p.stop)

    def test_deadline_armed_and_verified_before_detached_pilot(self):
        record = launch()
        self.assertEqual(record["state"], "started")
        calls = [call.args[0] for call in self.mocks[3].call_args_list]
        self.assertIn("--on-active=120m", calls[1])
        self.assertIn("is-active", calls[2])
        self.assertIn("--property=Type=exec", calls[3])
        self.assertTrue(calls[3][-1].endswith("run_first_arm_then_stop.sh"))

    def test_terminate_behavior_refuses_to_arm_shutdown(self):
        self.mocks[4].side_effect = None
        self.mocks[4].return_value = "terminate\n"
        with self.assertRaisesRegex(RuntimeError, "behavior set to stop"):
            launch()
        self.assertEqual(self.mocks[3].call_count, 1)  # only the privilege preflight

    def test_no_duplicate_pilot(self):
        self.mocks[4].side_effect = ["stop\n", "existing pilot service"]
        with self.assertRaisesRegex(RuntimeError, "already active"):
            launch()
        self.assertEqual(self.mocks[3].call_count, 1)

    def test_failed_service_launch_requests_immediate_shutdown(self):
        def execute(command, **kwargs):
            if "--property=Type=exec" in command:
                raise subprocess.CalledProcessError(1, command)
        self.mocks[3].side_effect = execute
        with self.assertRaises(subprocess.CalledProcessError):
            launch()
        last = self.mocks[3].call_args.args[0]
        self.assertEqual(last, ["sudo", "-n", "/usr/bin/shutdown", "-h", "now"])

    def test_runtime_limit_checked_before_host_access(self):
        for minutes in (0, 121, True):
            with self.assertRaises(ValueError):
                launch(minutes)
        self.mocks[0].assert_not_called()


if __name__ == "__main__":
    unittest.main()
