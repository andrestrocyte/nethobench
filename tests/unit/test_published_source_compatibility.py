"""Fail if an engineering change silently alters the paper's implementation."""
import hashlib
import json
from pathlib import Path


def test_published_package_sources_are_unchanged():
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "tests/resources/published_source_hashes.json").read_text())
    assert manifest["commit"] == "4075d2fe13b354de910d0cd1bb22826b94296594"
    for name, expected in manifest["sha256"].items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert actual == expected, f"Published implementation changed: {name}"
