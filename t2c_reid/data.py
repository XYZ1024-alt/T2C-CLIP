"""Dataset protocol helpers for Market-1501 and MSMT17."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

MARKET_PATTERN = re.compile(r"^(?P<pid>-?\d+)_c(?P<cam>\d+)s\d+_\d+_\d+\.jpg$")
MSMT_PATTERN = re.compile(r"^\d{4}_\d{3}_(?P<cam>\d{2})_.+\.jpg$")

MARKET_SPLIT_DIRS = {
    "train": "bounding_box_train",
    "query": "query",
    "gallery": "bounding_box_test",
}

MSMT_MANIFESTS = {
    "train": ("list_train.txt", "train"),
    "val": ("list_val.txt", "train"),
    "query": ("list_query.txt", "test"),
    "gallery": ("list_gallery.txt", "test"),
}

JUNK_PERSON_ID = -1


@dataclass(frozen=True)
class ReIDSample:
    image_path: Path
    person_id: int
    camera_id: int
    dataset: str
    split: str


def parse_market_filename(filename: str) -> tuple[int, int]:
    match = MARKET_PATTERN.match(Path(filename).name)
    if match is None:
        raise ValueError(f"Unsupported Market-1501 filename: {filename}")
    return int(match.group("pid")), int(match.group("cam"))


def parse_msmt17_filename(filename: str) -> int:
    match = MSMT_PATTERN.match(Path(filename).name)
    if match is None:
        raise ValueError(f"Unsupported MSMT17 filename: {filename}")
    return int(match.group("cam"))


def load_market_split(root: Path, split: str) -> list[ReIDSample]:
    split_dir = _market_split_dir(root, split)
    samples = [_market_sample(path, split) for path in sorted(split_dir.glob("*.jpg"))]
    return [sample for sample in samples if sample.person_id != JUNK_PERSON_ID]


def load_msmt17_manifest(root: Path, split: str) -> list[ReIDSample]:
    manifest_name, image_dir = _msmt_manifest(split)
    manifest_path = root / manifest_name
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    return [_msmt_sample(root, image_dir, split, line) for line in lines if line.strip()]


def _market_split_dir(root: Path, split: str) -> Path:
    if split not in MARKET_SPLIT_DIRS:
        raise ValueError(f"Unsupported Market-1501 split: {split}")
    return root / MARKET_SPLIT_DIRS[split]


def _market_sample(path: Path, split: str) -> ReIDSample:
    person_id, camera_id = parse_market_filename(path.name)
    return ReIDSample(path, person_id, camera_id, "market1501", split)


def _msmt_manifest(split: str) -> tuple[str, str]:
    if split not in MSMT_MANIFESTS:
        raise ValueError(f"Unsupported MSMT17 split: {split}")
    return MSMT_MANIFESTS[split]


def _msmt_sample(root: Path, image_dir: str, split: str, line: str) -> ReIDSample:
    relative_path, person_id = _parse_manifest_line(line)
    image_path = root / image_dir / relative_path
    camera_id = parse_msmt17_filename(relative_path.name)
    return ReIDSample(image_path, person_id, camera_id, "msmt17", split)


def _parse_manifest_line(line: str) -> tuple[Path, int]:
    parts = line.split()
    if len(parts) != 2:
        raise ValueError(f"Invalid MSMT17 manifest line: {line}")
    return Path(parts[0]), int(parts[1])
