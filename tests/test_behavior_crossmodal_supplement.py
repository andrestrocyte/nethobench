from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nethobench.analysis import behavior_crossmodal_supplement as supp
from nethobench.analysis import bulletproof_validation as bp
from nethobench.cli import main


def test_behavior_metrics_drop_for_time_shuffle() -> None:
    system = bp.generate_rich_system("low_rank_rnn", 7, n_sequences=4, seq_length=80, n_regions=8, latent_dim=5)
    candidates = supp.build_behavior_candidates(system.behavior, system.labels, seed=7)

    ceiling = supp.score_behavior_realism(system.behavior, system.labels, candidates["split_half_ceiling"].behavior, candidates["split_half_ceiling"].labels)
    shuffled = supp.score_behavior_realism(system.behavior, system.labels, candidates["time_shuffle"].behavior, candidates["time_shuffle"].labels)

    assert np.isfinite(ceiling["behavior_composite"])
    assert np.isfinite(shuffled["behavior_composite"])
    assert ceiling["behavior_composite"] > shuffled["behavior_composite"]
    assert ceiling["state_transition_score"] > shuffled["state_transition_score"]


def test_cross_modal_metrics_drop_under_misalignment() -> None:
    system = bp.generate_rich_system("nonlinear_lds", 11, n_sequences=4, seq_length=90, n_regions=8, latent_dim=5)
    behavior_candidates = supp.build_behavior_candidates(system.behavior, system.labels, seed=11)
    cross_candidates = supp.build_cross_modal_candidates(system, behavior_candidates, seed=11)

    aligned = supp.score_cross_modal_realism(system.neural, system.behavior, system.labels, cross_candidates["split_half_ceiling"].neural, cross_candidates["split_half_ceiling"].behavior, seed=11)
    lagged = supp.score_cross_modal_realism(system.neural, system.behavior, system.labels, cross_candidates["temporal_lag"].neural, cross_candidates["temporal_lag"].behavior, seed=11)

    assert np.isfinite(aligned["cross_modal_composite"])
    assert np.isfinite(lagged["cross_modal_composite"])
    assert aligned["cross_modal_composite"] > lagged["cross_modal_composite"]
    assert aligned["lagged_neural_behavior_score"] > lagged["lagged_neural_behavior_score"]


def test_behavior_crossmodal_cli_quick_outputs(tmp_path: Path) -> None:
    out = tmp_path / "behavior_cross"
    main(["behavior-crossmodal-supp", "--mode", "quick", "--output-root", str(out), "--min-free-disk-gb", "0.1"])

    expected = [
        "behavior_realism_scores.csv",
        "cross_modal_realism_scores.csv",
        "behavior_crossmodal_supplement_report.json",
        "behavior_crossmodal_supplement.svg",
    ]
    for name in expected:
        assert (out / name).is_file()

    behavior = pd.read_csv(out / "behavior_realism_scores.csv")
    cross = pd.read_csv(out / "cross_modal_realism_scores.csv")
    assert not behavior.empty
    assert not cross.empty
    assert np.isfinite(behavior["behavior_composite"]).all()
    assert np.isfinite(cross["cross_modal_composite"]).all()

    report = json.loads((out / "behavior_crossmodal_supplement_report.json").read_text())
    assert "NethoBench is implemented as a general structural-realism framework" in report["paper_interpretation"]
