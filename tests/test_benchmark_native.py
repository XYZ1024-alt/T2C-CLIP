import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from t2c_reid.cli.benchmark_native import _RssSampler, main


class NativeBenchmarkCliTest(unittest.TestCase):
    def test_tiny_benchmark_writes_structured_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "benchmark.json"
            exit_code = main(
                [
                    "--mode",
                    "all",
                    "--data-samples",
                    "4",
                    "--batch-size",
                    "2",
                    "--num-workers",
                    "0",
                    "--image-height",
                    "16",
                    "--image-width",
                    "8",
                    "--query-count",
                    "4",
                    "--gallery-count",
                    "8",
                    "--feature-dim",
                    "8",
                    "--rerank-query-count",
                    "3",
                    "--rerank-gallery-count",
                    "6",
                    "--chunk-size",
                    "2",
                    "--runs",
                    "1",
                    "--warmup-runs",
                    "0",
                    "--output",
                    str(output),
                ]
            )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(set(payload), {"data", "evaluation", "rerank", "settings"})
        for name in ("data", "evaluation"):
            self.assertIn("speedup", payload[name])
            self.assertIn("median_seconds", payload[name]["python"])
            self.assertIn("median_seconds", payload[name]["rust"])
        self.assertTrue(payload["evaluation"]["metrics_match"])
        self.assertTrue(payload["rerank"]["metrics_match"])
        self.assertIn("rss_ratio", payload["rerank"])
        self.assertIn("git_dirty", payload["settings"])
        self.assertIn("git_diff_sha256", payload["settings"])
        if payload["settings"]["git_diff_sha256"] is not None:
            self.assertEqual(len(payload["settings"]["git_diff_sha256"]), 64)

    def test_rss_sampler_propagates_background_probe_failure(self):
        with mock.patch(
            "t2c_reid.cli.benchmark_native._current_rss_bytes",
            side_effect=[100, RuntimeError("probe failed")],
        ):
            with self.assertRaisesRegex(RuntimeError, "RSS sampling failed"):
                with _RssSampler():
                    time.sleep(0.02)


if __name__ == "__main__":
    unittest.main()
