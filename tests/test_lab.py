"""Cloud CPU tests of measurement behavior; not GPU inference benchmarks."""
import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path

from aiohttp import web

from lab.analyze import extract_metrics, summarize
from lab.compare import load_matrix
from lab.collect import run, sse_data
from lab.workload import arrivals, schedule_records_digest, validate


def write_artifacts(root, cfg, policy, provenance):
    rows = [{"request_id": f"r{index}", "phase": phase["name"], "warmup": False,
             "prompt_tokens": phase["prompt_tokens"], "output_tokens": phase["output_tokens"],
             "status": "ok", "arrival_s": index*.05+.001, "actual_dispatch_s": index*.05+.001,
             "dispatch_lag_s": 0, "event_times_s": [index*.05+.002, index*.05+.003],
             "finished_s": index*.05+.004, "completion_tokens": phase["output_tokens"],
             "http_status": 200, "stream_done": True, "finish_reason": "length"}
            for index, phase in enumerate(cfg["phases"])]
    meta = {"config": cfg, "labels": {"policy": policy, "capacity": "roomy", "seed": 41},
            "provenance": copy.deepcopy(provenance), "complete": True, "scheduled_requests": len(rows),
            "recorded_requests": len(rows), "observed_duration_s": .2,
            "schedule_records_sha256": schedule_records_digest(rows)}
    (root/"run.json").write_text(json.dumps(meta))
    (root/"requests.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    (root/"metrics.jsonl").write_text("")
    (root/"summary.json").write_text(json.dumps({"complete": True}))
    return meta, rows


def config():
    data = json.loads(Path("configs/pilot.json").read_text())
    data["warmup_s"] = 0
    for phase in data["phases"]:
        phase.update(duration_s=.05, request_rate=50, prompt_tokens=3, output_tokens=2)
    data["request_timeout_s"] = 1
    data["drain_timeout_s"] = .1
    return data


class WorkloadTests(unittest.TestCase):
    def test_schedule_repeatability_and_boundaries(self):
        cfg = config()
        first = arrivals(cfg, 41)
        self.assertEqual(first, arrivals(cfg, 41))
        self.assertNotEqual(first, arrivals(cfg, 42))
        self.assertTrue(all(0 <= r["arrival_s"] < .1 for r in first))
        self.assertEqual(len(first), len({r["request_id"] for r in first}))

    def test_context_guard(self):
        cfg = config()
        cfg["phases"][0]["prompt_tokens"] = cfg["max_model_len"]
        with self.assertRaises(ValueError):
            validate(cfg)

    def test_no_silent_bad_rate(self):
        cfg = config()
        cfg["phases"][0]["request_rate"] = 0
        with self.assertRaises(ValueError):
            validate(cfg)

    def test_metric_aliases(self):
        parsed = extract_metrics('vllm:num_requests_running{model_name="x"} 3\n'
                                 'vllm:gpu_cache_usage_perc{model_name="x"} 0.8\n'
                                 'vllm:num_preemptions_total 4\n')
        self.assertEqual(parsed, {"running": 3, "kv_usage": .8, "preemptions": 4})

    def test_comparison_rejects_mixed_workloads_and_partial_matrix(self):
        cfg = config()
        provenance = {"workload": {"id": "frozen"}, "server": {"image_id": "sha256:fixture", "instance_id": "i-fixture"},
                      "schedule_sha256": "seed41"}
        with tempfile.TemporaryDirectory() as tmp:
            for idx, policy in enumerate(("fixed_2k", "fixed_16k")):
                root = Path(tmp)/str(idx)/"data"
                root.mkdir(parents=True)
                write_artifacts(root, cfg, policy, provenance)
            with self.assertRaisesRegex(ValueError, "Incomplete matrix"):
                load_matrix(tmp)
            runs, missing = load_matrix(tmp, allow_partial=True)
            self.assertEqual(len(runs), 2)
            self.assertEqual(len(missing), 16)
            path = Path(tmp)/"1/data/run.json"
            changed = json.loads(path.read_text())
            changed["provenance"]["schedule_sha256"] = "different"
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "Arrival schedule differs"):
                load_matrix(tmp, allow_partial=True)

    def test_failures_remain_in_slo_denominator(self):
        cfg = config()
        rows = [{"request_id": "a", "phase": "short_output", "warmup": False, "status": "ok",
                 "arrival_s": .001, "actual_dispatch_s": .001, "dispatch_lag_s": 0,
                 "event_times_s": [.002, .003], "finished_s": .004, "completion_tokens": 2},
                {"request_id": "b", "phase": "short_output", "warmup": False, "status": "timeout",
                 "arrival_s": .01, "actual_dispatch_s": .01, "dispatch_lag_s": 0,
                 "event_times_s": [], "finished_s": .11, "completion_tokens": None}]
        meta = {"config": cfg, "labels": {}, "complete": True, "scheduled_requests": 2, "observed_duration_s": .2}
        result = summarize(meta, rows, [])
        self.assertEqual(result["phases"]["short_output"]["slo_attainment_all_arrivals"], .5)
        self.assertEqual(result["completed_output_tokens_per_drain_inclusive_s"], 10)
        self.assertIn("sparse_p99_exploratory_only", result["phases"]["short_output"]["quality_flags"])


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.mode = "ok"

        async def health(req):
            return web.Response(text="ok")

        async def metrics(req):
            return web.Response(text="vllm:num_requests_running 0\nvllm:num_requests_waiting 0\nvllm:kv_cache_usage_perc 0.1\nvllm:num_preemptions_total 0\n")

        async def completion(req):
            body = await req.json()
            self.assertEqual(body["prompt"], [1, 2, 3])
            if self.mode == "http_error":
                return web.Response(status=503, text="busy")
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(req)
            if self.mode == "stall":
                await asyncio.sleep(.3)
                return resp
            for text in ("hé", "llo"):
                payload = ('data: ' + json.dumps({"choices": [{"text": text}]}, ensure_ascii=False) + '\r\n\r\n').encode()
                # Deliberately fragment transport, including UTF-8 and SSE separators.
                for fragment in (payload[:14], payload[14:23], payload[23:]):
                    await resp.write(fragment)
                await asyncio.sleep(.003)
            if self.mode != "truncated":
                payload = {"choices": [{"text": "", "finish_reason": "length"}],
                           "usage": {"completion_tokens": body["max_tokens"] if self.mode != "wrong_length" else 1}}
                await resp.write(('data: ' + json.dumps(payload) + '\n\ndata: [DONE]\n\n').encode())
            await resp.write_eof()
            return resp

        app = web.Application()
        app.router.add_get("/health", health)
        app.router.add_get("/metrics", metrics)
        app.router.add_post("/v1/completions", completion)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        port = self.site._server.sockets[0].getsockname()[1]
        self.endpoint = f"http://127.0.0.1:{port}"

    async def asyncTearDown(self):
        await self.runner.cleanup()

    async def exercise(self, mode, cap=None):
        self.mode = mode
        cfg = config()
        if cap:
            cfg["max_active_requests"] = cap
        schedule = [{"request_id": f"r{i}", "arrival_s": .005+i*.004, "phase": "short_output",
                     "warmup": False, "prompt_tokens": 3, "output_tokens": 2,
                     "prompt_token_ids": [1, 2, 3]} for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/"run"
            meta = await run(self.endpoint, schedule, cfg, target, {"fixture": True}, {"synthetic_test": True})
            rows = [json.loads(line) for line in (target/"requests.jsonl").read_text().splitlines()]
            self.assertTrue(meta["complete"])
            self.assertEqual(len(rows), 3)
            self.assertEqual(len({r["request_id"] for r in rows}), 3)
            return rows

    async def test_fragmented_stream_and_usage(self):
        rows = await self.exercise("ok")
        self.assertTrue(all(r["status"] == "ok" for r in rows))
        self.assertTrue(all(len(r["event_times_s"]) == 2 for r in rows))
        self.assertTrue(all(r["completion_tokens"] == 2 for r in rows))

    async def test_truncated_stream_not_success(self):
        self.assertTrue(all(r["status"] == "truncated" for r in await self.exercise("truncated")))

    async def test_usage_mismatch_not_success(self):
        self.assertTrue(all(r["status"] == "length_mismatch" for r in await self.exercise("wrong_length")))

    async def test_http_errors_accounted(self):
        self.assertTrue(all(r["status"] == "http_error" for r in await self.exercise("http_error")))

    async def test_censored_requests_accounted(self):
        self.assertTrue(all(r["status"] == "censored" for r in await self.exercise("stall")))

    async def test_capacity_guard_is_explicit(self):
        rows = await self.exercise("stall", cap=1)
        self.assertEqual(sum(r["status"] == "client_capacity_rejected" for r in rows), 2)


if __name__ == "__main__":
    unittest.main()
