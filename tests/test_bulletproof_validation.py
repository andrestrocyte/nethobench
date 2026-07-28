from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nethobench.analysis import bulletproof_validation as bp
from nethobench.cli import main


def test_rich_generators_are_deterministic_and_well_shaped() -> None:
    systems = ["nonlinear_lds", "switching_lds", "lorenz", "low_rank_rnn", "spiking_calcium"]
    for name in systems:
        a = bp.generate_rich_system(name, 7, n_sequences=3, seq_length=60, n_regions=6, latent_dim=4)
        b = bp.generate_rich_system(name, 7, n_sequences=3, seq_length=60, n_regions=6, latent_dim=4)
        assert a.neural.shape == (3, 60, 6)
        assert a.oracle.shape == (3, 60, 6)
        assert a.labels.shape == (3, 60)
        assert a.behavior.shape[:2] == (3, 60)
        assert np.isfinite(a.neural).all()
        np.testing.assert_allclose(a.neural, b.neural)


def test_slds_exports_known_transition_matrix() -> None:
    system = bp.generate_rich_system("switching_lds", 11, n_sequences=5, seq_length=80, n_regions=7, latent_dim=4)
    transition = np.asarray(system.metadata["transition_matrix"], dtype=float)
    assert transition.shape == (3, 3)
    np.testing.assert_allclose(transition.sum(axis=1), 1.0)
    empirical = bp._state_transition_matrix(system.labels)
    assert empirical.shape == transition.shape
    assert np.trace(empirical) > 2.0


def test_wrong_dynamics_controls_are_mse_matched_but_realism_separates() -> None:
    system = bp.generate_rich_system("low_rank_rnn", 19, n_sequences=4, seq_length=80, n_regions=8, latent_dim=5)
    controls = bp.build_controls(system, seed=19, target_mse=0.20)
    selected = ["smoothed_mean", "short_horizon_good", "transition_shuffle"]
    mses = [np.mean((system.neural - controls[name]) ** 2) for name in selected]
    assert max(mses) - min(mses) < 0.025
    scores = {name: bp.score_arrays_fast(system.neural, controls[name])["FINAL_COMPOSITE_SCORE"] for name in selected}
    assert max(scores.values()) - min(scores.values()) > 0.02


def test_behavior_and_perturbation_validity_controls_have_expected_ordering() -> None:
    system = bp.generate_rich_system("nonlinear_lds", 7, n_sequences=4, seq_length=90, n_regions=8, latent_dim=5)
    controls = bp.build_controls(system, seed=7, target_mse=0.20)
    oracle = bp.task_behavior_validity(system, controls["oracle"], candidate_name="oracle")
    shuffled = bp.task_behavior_validity(system, controls["marginal_shuffle"], candidate_name="marginal_shuffle")
    assert oracle["decoding_preservation"] >= shuffled["decoding_preservation"]

    rows = pd.DataFrame(bp.perturbation_validation(system, {"oracle": controls["oracle"], "transition_shuffle": controls["transition_shuffle"]}))
    assert {"response_corr", "peak_latency_score", "amplitude_score"}.issubset(rows.columns)
    assert rows.loc[rows["candidate"] == "oracle", "amplitude_score"].iloc[0] >= 0.0


def test_synthetic_bold_shape_and_ceiling_floor() -> None:
    system = bp.generate_rich_system("nonlinear_lds", 7, n_sequences=3, seq_length=80, n_regions=6, latent_dim=4)
    bold = bp.neural_to_synthetic_bold(system.neural, tr_bins=4, seed=0)
    assert bold.shape == (3, 20, 6)
    rows = pd.DataFrame(bp._score_bold(system, bp.build_controls(system, seed=7, target_mse=0.20)))
    ceiling = rows.loc[rows["candidate"] == "bold_split_half_ceiling", "FINAL_COMPOSITE_SCORE"].iloc[0]
    floor = rows.loc[rows["candidate"] == "bold_time_parcel_shuffle_floor", "FINAL_COMPOSITE_SCORE"].iloc[0]
    assert ceiling > floor


def test_disk_guard_and_cli_quick_mode(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        bp.run_bulletproof_validation(output_root=tmp_path / "too_strict", mode="quick", min_free_disk_gb=1_000_000)

    out = tmp_path / "bulletproof"
    main(["bulletproof-run", "--mode", "quick", "--output-root", str(out), "--min-free-disk-gb", "0.1"])
    expected = [
        "rich_synthetic_scores_long.csv",
        "wrong_dynamics_controls.csv",
        "task_behavior_validity_scores.csv",
        "perturbation_response_scores.csv",
        "fmri_bold_validation_scores.csv",
        "bulletproof_analysis_report.json",
        "bulletproof_validation_summary.svg",
        "same_mse_different_realism.svg",
    ]
    for name in expected:
        assert (out / name).is_file()
    report = json.loads((out / "bulletproof_analysis_report.json").read_text())
    assert len(report["systems"]) >= 4
    assert report["disk_events"][0]["free_gb"] >= report["disk_events"][0]["min_free_gb"]
