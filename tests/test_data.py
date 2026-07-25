from pathlib import Path
import tempfile
import unittest

from t2c_reid.data import (
    ReIDSample,
    load_market_split,
    load_msmt17_manifest,
    parse_market_filename,
    parse_msmt17_filename,
)


class DataParsingTest(unittest.TestCase):
    def test_parse_market_filename_reads_person_and_camera(self):
        self.assertEqual(parse_market_filename("0002_c3s1_000551_01.jpg"), (2, 3))

    def test_parse_msmt17_filename_reads_camera_token(self):
        self.assertEqual(parse_msmt17_filename("0000_045_12_0303morning_0006_2.jpg"), 12)

    def test_load_market_split_skips_junk_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            split_dir = Path(tmp) / "bounding_box_train"
            split_dir.mkdir()
            (split_dir / "0002_c3s1_000551_01.jpg").touch()
            (split_dir / "-1_c1s1_000401_03.jpg").touch()

            samples = load_market_split(Path(tmp), "train")

        self.assertEqual(
            samples,
            [ReIDSample(split_dir / "0002_c3s1_000551_01.jpg", 2, 3, "market1501", "train")],
        )

    def test_load_msmt17_manifest_uses_manifest_label_and_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "list_query.txt").write_text(
                "0000/0000_000_01_0303morning_0015_0.jpg 7\n",
                encoding="utf-8",
            )

            samples = load_msmt17_manifest(root, "query")

        expected_path = root / "test" / "0000" / "0000_000_01_0303morning_0015_0.jpg"
        self.assertEqual(samples, [ReIDSample(expected_path, 7, 1, "msmt17", "query")])
