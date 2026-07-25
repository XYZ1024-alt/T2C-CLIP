import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from t2c_reid.cli.evaluate import main


class EvaluateCliTest(unittest.TestCase):
    def test_main_writes_no_rerank_metrics_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "features.npz"
            output_path = Path(tmp) / "metrics.json"
            np.savez(
                input_path,
                query_features=np.array([[1.0, 0.0]], dtype=np.float32),
                gallery_features=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                query_ids=np.array([1], dtype=np.int64),
                gallery_ids=np.array([2, 1], dtype=np.int64),
                query_cams=np.array([1], dtype=np.int64),
                gallery_cams=np.array([2, 2], dtype=np.int64),
            )

            exit_code = main([str(input_path), "--output", str(output_path), "--ranks", "1"])
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload, {"cmc": {"1": 0.0}, "mAP": 0.5})

    def test_main_appends_rerank_metrics_without_replacing_primary(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "features.npz"
            output_path = Path(tmp) / "metrics.json"
            np.savez(
                input_path,
                query_features=np.array([[1.0, 0.0]], dtype=np.float32),
                gallery_features=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                query_ids=np.array([1], dtype=np.int64),
                gallery_ids=np.array([2, 1], dtype=np.int64),
                query_cams=np.array([1], dtype=np.int64),
                gallery_cams=np.array([2, 2], dtype=np.int64),
            )

            exit_code = main(
                [
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--ranks",
                    "1",
                    "--report-rerank",
                    "--rerank-k1",
                    "1",
                    "--rerank-k2",
                    "1",
                ]
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["mAP"], 0.5)
        self.assertEqual(payload["cmc"], {"1": 0.0})
        self.assertIn("rerank", payload)
        self.assertIn("mAP", payload["rerank"])
        self.assertEqual(set(payload["rerank"]), {"cmc", "mAP"})
