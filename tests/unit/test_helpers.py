from __future__ import annotations

import pandas as pd

from nethobench.utils.helpers import load_gt_and_preds


def _pose_frame(offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequenceId": [0, 0, 1, 1],
            "itemPosition": [0, 1, 0, 1],
            "CENTER_X": [0.0 + offset, 1.0 + offset, 2.0 + offset, 3.0 + offset],
            "CENTER_Y": [0.0, 1.0, 2.0, 3.0],
            "NOSE_X": [0.1, 1.1, 2.1, 3.1],
            "NOSE_Y": [0.0, 1.0, 2.0, 3.0],
            "TAIL_BASE_X": [-0.1, 0.9, 1.9, 2.9],
            "TAIL_BASE_Y": [0.0, 1.0, 2.0, 3.0],
        }
    )


def test_load_gt_and_preds_pairs_single_gt_with_prediction_directory(tmp_path):
    gt_path = tmp_path / "gt.csv"
    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()

    _pose_frame().to_csv(gt_path, index=False)
    _pose_frame(offset=0.1).to_csv(pred_dir / "pred_a.csv", index=False)
    _pose_frame(offset=0.2).to_csv(pred_dir / "pred_b.csv", index=False)

    gt_df, pred_df = load_gt_and_preds(gt_path, pred_dir)

    assert len(gt_df) == 8
    assert len(pred_df) == 8
    assert not gt_df.duplicated(["sequenceId", "itemPosition"]).any()
    assert not pred_df.duplicated(["sequenceId", "itemPosition"]).any()
    assert set(gt_df["sequenceId"]) == {"0:0", "0:1", "1:0", "1:1"}
