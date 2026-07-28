from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nethobench.tribev2_adapter import (
    SEQUENCE_KEY,
    TIME_KEY,
    _apply_corruption,
    _compute_score_bundle,
    _frame_from_array,
    write_manifest_template,
)


def _make_frame(sequence_id: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_time = 96
    n_regions = 16
    time = np.linspace(0.0, 6.0 * np.pi, n_time)
    base = np.stack(
        [
            np.sin(time * (0.35 + (0.03 * idx))) + (0.15 * np.cos(time * (0.12 + (0.01 * idx))))
            for idx in range(n_regions)
        ],
        axis=1,
    )
    noise = rng.normal(scale=0.08, size=base.shape)
    arr = base + noise
    cols = [f"parcel_{idx:02d}" for idx in range(n_regions)]
    return _frame_from_array(sequence_id, arr, cols)


def test_write_manifest_template(tmp_path: Path) -> None:
    out = tmp_path / "tribe_manifest.json"
    write_manifest_template("hcp", out, name="hcp-test", tribev2_root="/tmp/tribe")
    payload = json.loads(out.read_text())
    assert payload["name"] == "hcp-test"
    assert payload["dataset"] == "HCP"
    assert payload["tribev2_root"] == "/tmp/tribe"
    assert payload["stimuli"][0]["sequence_id"] == "hcp_movie"


def test_compute_score_bundle_and_corruption_shapes() -> None:
    parcel_columns = [f"parcel_{idx:02d}" for idx in range(16)]
    gt = pd.concat([_make_frame("seq_a", 1), _make_frame("seq_b", 2)], ignore_index=True)
    pred = gt.copy()
    pred.loc[:, parcel_columns] = pred[parcel_columns].to_numpy() + 0.05

    bundle = _compute_score_bundle(gt, pred, parcel_columns)
    assert "neuro_scores" in bundle
    assert "fidelity_scores" in bundle
    assert "FINAL_COMPOSITE_SCORE" in bundle["neuro_scores"]
    assert "FIDELITY_SCORE" in bundle["fidelity_scores"]

    corrupted = _apply_corruption(pred, parcel_columns, "region_permute", level=1.0, seed=7)
    assert list(corrupted.columns) == list(pred.columns)
    assert corrupted.shape == pred.shape
    assert corrupted[SEQUENCE_KEY].tolist() == pred[SEQUENCE_KEY].tolist()
    assert corrupted[TIME_KEY].tolist() == pred[TIME_KEY].tolist()
