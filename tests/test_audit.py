"""Regression tests for data loss, false success and comparison corruption."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from lab.analyze import summarize
from lab.audit import audit, verify_records
from lab.compare import load_matrix
from lab.workload import digest, schedule_records_digest
from test_lab import config, write_artifacts


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)/"arm"/"data"
        self.root.mkdir(parents=True)
        self.cfg = config()
        self.provenance = {"workload": {"id": "frozen"},
                           "server": {"image_id": "sha256:fixture", "instance_id": "i-fixture"},
                           "schedule_sha256": "seed41", "execution_kind": "cpu_fixture"}
        self.meta, self.rows = write_artifacts(self.root, self.cfg, "fixed_2k", self.provenance)

    def test_detects_missing_or_replaced_request(self):
        with self.assertRaisesRegex(ValueError, "Incomplete request accounting"):
            verify_records(self.meta, self.rows[:-1], [])
        changed = copy.deepcopy(self.rows)
        changed[0]["request_id"] = "replacement"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            verify_records(self.meta, changed, [])

    def test_detects_impossible_stream_or_false_success(self):
        changed = copy.deepcopy(self.rows)
        changed[0]["event_times_s"].reverse()
        with self.assertRaisesRegex(ValueError, "out of order"):
            verify_records(self.meta, changed, [])
        for key, value in (("completion_tokens", 1), ("stream_done", False), ("finish_reason", None)):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.rows)
                changed[0][key] = value
                with self.assertRaisesRegex(ValueError, "Successful request"):
                    verify_records(self.meta, changed, [])

    def test_bad_metrics_are_rejected(self):
        for metrics in ([{"t_s": float("nan")}],
                        [{"t_s": .01, "raw_prometheus": "vllm:kv_cache_usage_perc 1.2\n"}]):
            with self.subTest(metrics=metrics), self.assertRaises(ValueError):
                verify_records(self.meta, self.rows, metrics)

    def test_comparison_recomputes_stale_summary_from_raw_records(self):
        (self.root/"summary.json").write_text(json.dumps({"complete": False, "labels": {"policy": "invented"}}))
        runs, _ = load_matrix(self.directory.name, allow_partial=True)
        self.assertTrue(runs[0][0]["complete"])
        self.assertEqual(runs[0][0]["labels"]["policy"], "fixed_2k")
        self.assertEqual(runs[0][0]["record_count"], 2)
        self.rows[0]["completion_tokens"] = 1
        (self.root/"requests.jsonl").write_text("".join(json.dumps(row)+"\n" for row in self.rows))
        with self.assertRaisesRegex(ValueError, "Successful request"):
            load_matrix(self.directory.name, allow_partial=True)

    def test_pressure_is_reported_per_phase_with_sample_counts(self):
        metrics = [{"t_s": tick, "raw_prometheus": f"vllm:kv_cache_usage_perc {kv}\nvllm:num_preemptions_total {counter}\n"}
                   for tick, kv, counter in ((.002, .2, 0), (.03, .95, 1), (.051, .9, 1), (.09, .99, 4))]
        result = summarize(self.meta, self.rows, metrics)
        short = result["phases"]["short_output"]
        long = result["phases"]["long_output"]
        self.assertEqual(short["kv_usage_sample_count"], 2)
        self.assertEqual(short["kv_samples_at_or_above_90pct_fraction"], .5)
        self.assertEqual(long["kv_usage_peak_fraction"], .99)
        self.assertEqual(long["preemption_delta_between_in_phase_samples"], 3)

    def test_frozen_schedule_token_corruption_is_detected(self):
        workload = Path(self.directory.name)/"workload"
        workload.mkdir()
        (workload/"config.json").write_text(json.dumps(self.cfg, indent=2))
        scheduled = [{key: row[key] for key in ("request_id", "phase", "warmup", "prompt_tokens", "output_tokens", "arrival_s")}
                     for row in self.rows]
        for row in scheduled:
            row["prompt_token_ids"] = [1, 2, 3]
        schedule = workload/"seed-41.jsonl"
        schedule.write_text("".join(json.dumps(row)+"\n" for row in scheduled))
        manifest = {"config_sha256": digest(workload/"config.json"), "model_revision": "0"*40,
                    "schedule_sha256": {schedule.name: digest(schedule)}}
        (workload/"manifest.json").write_text(json.dumps(manifest))
        self.meta["provenance"] = {"workload": manifest, "schedule_sha256": digest(schedule),
            "server": {"model": self.cfg["model"], "model_revision": manifest["model_revision"],
                       "policy": "fixed_2k", "capacity": "roomy", "kv_capacity_bytes": self.cfg["kv_capacities_bytes"]["roomy"]}}
        verify_records(self.meta, self.rows, [], workload)
        scheduled[0]["prompt_token_ids"][0] = 99
        schedule.write_text("".join(json.dumps(row)+"\n" for row in scheduled))
        with self.assertRaisesRegex(ValueError, "Frozen schedule hash mismatch"):
            verify_records(self.meta, self.rows, [], workload)

    def test_comparison_rejects_fixture_and_gpu_mix(self):
        root = Path(self.directory.name)/"second"/"data"
        root.mkdir(parents=True)
        provenance = copy.deepcopy(self.provenance)
        provenance["execution_kind"] = "ec2_gpu"
        write_artifacts(root, self.cfg, "fixed_16k", provenance)
        with self.assertRaisesRegex(ValueError, "Execution kinds differ"):
            load_matrix(self.directory.name, allow_partial=True)

    def test_pilot_failure_gate_preserves_failure_outcomes_for_full_study(self):
        metrics = [{"t_s": tick, "raw_prometheus": "vllm:kv_cache_usage_perc 0.2\nvllm:num_preemptions_total 0\n"
                    "vllm:num_requests_running 1\nvllm:num_requests_waiting 0\n"}
                   for tick in (.002, .03, .051, .09)]
        (self.root/"metrics.jsonl").write_text("".join(json.dumps(row)+"\n" for row in metrics))
        self.assertTrue(audit(self.root, pilot=True)["ready"])
        self.rows[0]["status"] = "timeout"
        (self.root/"requests.jsonl").write_text("".join(json.dumps(row)+"\n" for row in self.rows))
        self.assertTrue(audit(self.root)["ready"])
        report = audit(self.root, pilot=True)
        self.assertTrue(report["artifact_valid"])
        self.assertFalse(report["ready"])
        self.assertIn("pilot_has_unsuccessful_requests", report["blockers"])

    def test_pilot_requires_pressure_samples_in_each_phase(self):
        metrics = [{"t_s": .002, "raw_prometheus": "vllm:kv_cache_usage_perc 0.2\nvllm:num_preemptions_total 0\n"
                    "vllm:num_requests_running 1\nvllm:num_requests_waiting 0\n"}]
        (self.root/"metrics.jsonl").write_text(json.dumps(metrics[0])+"\n")
        report = audit(self.root, pilot=True)
        self.assertFalse(report["ready"])
        self.assertIn("long_output:no_phase_kv_samples", report["blockers"])


if __name__ == "__main__":
    unittest.main()
