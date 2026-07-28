from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import nethobench.ibl_adapter as ibl
from nethobench.cli import main


def test_ibl_cli_manifest_and_dry_run(tmp_path: Path) -> None:
    manifest_path = tmp_path / "ibl_manifest.json"
    main(["ibl-init-manifest", "--output", str(manifest_path)])

    payload = json.loads(manifest_path.read_text())
    assert payload["base_url"] == "https://openalyx.internationalbrainlab.org"
    assert payload["release_tag"] == "RepeatedSite"
    assert payload["context_bins"] + payload["target_bins"] == 100

    output_root = tmp_path / "dry_run"
    main(["ibl-export", "--manifest", str(manifest_path), "--output-root", str(output_root), "--dry-run"])
    report = json.loads((output_root / "ibl_repeated_site_report.json").read_text())
    assert report["dry_run"] is True
    assert "gt_region_rates.csv" in report["expected_outputs"]


def test_filter_bin_and_region_aggregation_are_deterministic() -> None:
    spike_times = np.asarray([0.05, 0.15, 0.18, 1.05, 1.12, 1.16], dtype=float)
    spike_clusters = np.asarray([0, 0, 1, 2, 2, 2], dtype=int)
    clusters = {
        "label": np.asarray([1, 0, 1]),
        "acronym": np.asarray(["CA1", "VISa", "LP"], dtype=object),
    }

    units = ibl.filter_good_units(
        spike_times,
        spike_clusters,
        clusters,
        min_fr_hz=0.1,
        max_fr_hz=100.0,
        cluster_label_keep=[1],
    )
    assert units["cluster_id"].tolist() == [0, 2]
    assert units["region"].tolist() == ["hippocampus", "thalamus"]

    counts = ibl.bin_spike_counts(
        spike_times,
        spike_clusters,
        units["cluster_id"],
        np.asarray([0.0, 1.0]),
        window_seconds=(0.0, 0.2),
        bin_size_seconds=0.1,
    )
    assert counts.shape == (2, 2, 2)
    assert counts[0, :, 0].tolist() == [1.0, 1.0]
    assert counts[1, :, 1].tolist() == [1.0, 2.0]

    rates, regions = ibl.aggregate_region_rates(counts, units, bin_size_seconds=0.1)
    shuffled_rates, shuffled_regions = ibl.aggregate_region_rates(
        counts[:, :, [1, 0]],
        units.iloc[[1, 0]].reset_index(drop=True),
        bin_size_seconds=0.1,
    )
    assert regions == ["hippocampus", "thalamus"]
    assert shuffled_regions == regions
    np.testing.assert_allclose(shuffled_rates, rates)


def test_region_panel_zscore_frame_baselines_and_corruption() -> None:
    panel = ibl.choose_region_panel(
        [
            {"hippocampus": 6, "thalamus": 7, "posterior_parietal": 5},
            {"hippocampus": 8, "thalamus": 7, "posterior_parietal": 1},
        ],
        target_n_regions=2,
        min_n_regions=2,
        min_units_per_region=5,
    )
    assert panel == ["hippocampus", "thalamus"]
    with np.testing.assert_raises(ValueError):
        ibl.choose_region_panel(
            [
                {"hippocampus": 6, "thalamus": 7},
                {"hippocampus": 8, "posterior_parietal": 8},
            ],
            target_n_regions=2,
            min_n_regions=2,
            min_units_per_region=5,
        )

    rng = np.random.default_rng(11)
    arr = rng.normal(size=(6, 10, 3))
    norm, mean, std = ibl.zscore_from_train_context(arr, [0, 1, 2], context_bins=4)
    assert norm.shape == arr.shape
    np.testing.assert_allclose(norm[[0, 1, 2], :4].reshape(-1, 3).mean(axis=0), 0.0, atol=1e-12)
    assert np.all(std > 0)
    assert mean.shape == (3,)

    poisson = ibl.poisson_psth_prediction(norm[:3], n_sequences=2)
    var_pred = ibl.var_rollout_prediction(norm[:3], norm[3:5].copy(), context_bins=4)
    assert poisson.shape == (2, 10, 3)
    assert var_pred.shape == (2, 10, 3)
    np.testing.assert_allclose(var_pred[:, :4, :], norm[3:5, :4, :])

    corrupted = ibl.apply_ibl_corruption(norm, "channel_permute", level=1.0, seed=3)
    assert corrupted.shape == norm.shape

    frame = ibl.array_to_nethobench_frame(norm[:2], ["a", "b", "c"], sequence_prefix="ibl")
    assert frame.shape == (20, 5)
    assert frame.columns.tolist() == ["sequenceId", "itemPosition", "a", "b", "c"]


def test_evaluate_region_dataset_writes_standard_artifacts(tmp_path: Path, monkeypatch) -> None:
    def fake_score(gt: np.ndarray, pred: np.ndarray, channels: list[str]) -> dict[str, float]:
        assert gt.shape == pred.shape
        assert gt.shape[2] == len(channels)
        return {
            "family_distribution": 0.7,
            "family_temporal_spectral": 0.6,
            "family_relational": 0.5,
            "family_geometry": 0.8,
            "family_state_dynamics": 0.4,
            "FINAL_COMPOSITE_SCORE": 0.6,
        }

    def fake_fidelity(gt: np.ndarray, pred: np.ndarray, channels: list[str]) -> dict[str, float]:
        return {"FIDELITY_SCORE": 0.5}

    monkeypatch.setattr(ibl, "_score_arrays", fake_score)
    monkeypatch.setattr(ibl, "_fidelity_arrays", fake_fidelity)

    rng = np.random.default_rng(5)
    arr = rng.normal(size=(10, 10, 8))
    channels = [f"region_{idx}" for idx in range(8)]
    manifest = ibl.IBLManifest(
        window_seconds=(0.0, 1.0),
        bin_size_seconds=0.1,
        context_bins=4,
        target_bins=6,
    )
    report = ibl._evaluate_region_dataset(arr, channels, manifest, output_root=tmp_path)

    outputs = report["outputs"]
    assert Path(outputs["gt_region_rates_csv"]).is_file()
    assert Path(outputs["pred_poisson_region_rates_csv"]).is_file()
    assert Path(outputs["pred_var_region_rates_csv"]).is_file()
    assert Path(outputs["ibl_region_config_json"]).is_file()
    assert Path(outputs["ibl_split_half_ceiling_scores_json"]).is_file()
    assert Path(outputs["ibl_model_scores_json"]).is_file()
    assert Path(outputs["ibl_corruption_ladder_scores_json"]).is_file()
    assert Path(outputs["family_comparison_plot"]).is_file()
    assert Path(outputs["corruption_ladder_plot"]).is_file()

    gt_frame = pd.read_csv(outputs["gt_region_rates_csv"])
    assert {"sequenceId", "itemPosition", *channels}.issubset(gt_frame.columns)


def test_expanded_study_dry_run_cli(tmp_path: Path) -> None:
    manifest_path = tmp_path / "ibl_manifest.json"
    main(["ibl-init-manifest", "--output", str(manifest_path)])

    for command in ["ibl-build-dataset", "ibl-train-models", "ibl-score-study", "ibl-run-study"]:
        output_root = tmp_path / command
        main([command, "--manifest", str(manifest_path), "--output-root", str(output_root), "--dry-run"])
        assert any(output_root.glob("*.json"))


def test_raw_cache_cleanup_removes_only_new_large_spike_files(tmp_path: Path, monkeypatch) -> None:
    cache = tmp_path / "ONE"
    old_dir = cache / "old"
    new_dir = cache / "new"
    old_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    old_file = old_dir / "spikes.times.npy"
    old_file.write_bytes(b"0" * 11_000_000)
    small_new = new_dir / "spikes.clusters.npy"
    small_new.write_bytes(b"0" * 10)
    monkeypatch.setattr(ibl, "_one_cache_root", lambda: cache)
    before = ibl._snapshot_large_one_files()
    new_file = new_dir / "spikes.times.npy"
    new_file.write_bytes(b"1" * 11_000_000)

    removed = ibl._clean_owned_raw_files(before, enabled=True)

    assert str(new_file) in removed
    assert old_file.exists()
    assert small_new.exists()
    assert not new_file.exists()


def test_expanded_study_training_and_scoring_on_synthetic_features(tmp_path: Path, monkeypatch) -> None:
    def fake_score(gt: np.ndarray, pred: np.ndarray, channels: list[str]) -> dict[str, float]:
        assert gt.shape == pred.shape
        return {
            "family_distribution": 0.7,
            "family_temporal_spectral": 0.6,
            "family_relational": 0.5,
            "family_geometry": 0.8,
            "family_state_dynamics": 0.4,
            "FINAL_COMPOSITE_SCORE": 0.6,
        }

    def fake_fidelity(gt: np.ndarray, pred: np.ndarray, channels: list[str]) -> dict[str, float]:
        return {"FIDELITY_SCORE": 0.5}

    def fake_torch_model(
        model_name,
        train,
        val,
        test,
        manifest,
        *,
        seed,
        checkpoint_path=None,
    ):
        return test * 0.95, {
            "seed": seed,
            "fake": True,
            "checkpoint_path": (
                str(checkpoint_path) if checkpoint_path is not None else None
            ),
        }

    def fake_ssm_vae_model(
        train,
        val,
        test,
        manifest,
        *,
        seed,
        checkpoint_path=None,
    ):
        samples = np.repeat((test * 0.9)[None, :, :, :], int(manifest.ssm_vae_samples), axis=0)
        return test * 0.9, samples, {
            "seed": seed,
            "fake": True,
            "n_samples": int(manifest.ssm_vae_samples),
            "checkpoint_path": (
                str(checkpoint_path) if checkpoint_path is not None else None
            ),
        }

    monkeypatch.setattr(ibl, "_score_arrays", fake_score)
    monkeypatch.setattr(ibl, "_fidelity_arrays", fake_fidelity)
    monkeypatch.setattr(ibl, "_train_torch_forecaster", fake_torch_model)
    monkeypatch.setattr(ibl, "_train_ssm_vae_forecaster", fake_ssm_vae_model)

    rng = np.random.default_rng(17)
    arr = np.log1p(rng.poisson(2.0, size=(24, 10, 4)).astype(float))
    regions = ["hippocampus", "thalamus", "ssptr", "vispa"]
    region_npz = tmp_path / "ibl_region_rates_raw.npz"
    np.savez_compressed(region_npz, region_rates=arr.astype(np.float32), regions=np.asarray(regions, dtype=object))
    sequence_table = pd.DataFrame(
        {
            "sequence_index": np.arange(arr.shape[0]),
            "eid": np.repeat(["s1", "s2", "s3", "s4"], 6),
            "trial_index": np.tile(np.arange(6), 4),
            "lab": np.repeat(["l1", "l2", "l3", "l4"], 6),
            "subject": np.repeat(["a", "b", "c", "d"], 6),
        }
    )
    sequence_path = tmp_path / "ibl_sequence_table.csv"
    sequence_table.to_csv(sequence_path, index=False)
    inventory_path = tmp_path / "ibl_dataset_inventory.csv"
    sequence_table.groupby(["eid", "lab", "subject"]).size().reset_index(name="n").to_csv(inventory_path, index=False)
    feature_manifest = {
        "region_rates_npz": str(region_npz),
        "sequence_table_csv": str(sequence_path),
        "inventory_csv": str(inventory_path),
        "region_panel": regions,
    }
    (tmp_path / "ibl_feature_manifest.json").write_text(json.dumps(feature_manifest))
    manifest_path = tmp_path / "manifest.json"
    manifest = ibl.IBLManifest(
        window_seconds=(0.0, 1.0),
        bin_size_seconds=0.1,
        context_bins=4,
        target_bins=6,
        model_suite=["psth", "glm_poisson", "var", "lds_var", "ssm_vae", "gru", "transformer", "transformer_nb_reg"],
        neural_seeds=[7],
        ssm_vae_samples=3,
        max_epochs=1,
        patience=1,
        bootstrap_samples=5,
        output_root=str(tmp_path),
    )
    manifest_path.write_text(json.dumps(ibl._ibl_json_ready(ibl.asdict(manifest)), indent=2))

    train_report = ibl.train_ibl_study_models(manifest_path, output_root=tmp_path)
    assert Path(train_report["predictions_manifest_json"]).is_file()
    score_report = ibl.score_ibl_study(manifest_path, output_root=tmp_path)
    outputs = score_report["report"]["outputs"]
    assert Path(outputs["ibl_model_scores_long_csv"]).is_file()
    assert Path(outputs["ibl_bootstrap_ci_json"]).is_file()
    assert Path(outputs["family_comparison_plot"]).is_file()
    assert Path(outputs["model_family_heatmap_plot"]).is_file()
    pred_manifest = json.loads(Path(train_report["predictions_manifest_json"]).read_text())
    within_models = pred_manifest["tasks"]["within_session"]["models"]
    assert "stochastic_prediction_csv" in within_models["ssm_vae"]
    assert Path(outputs["ssm_vae_diagnostics_json"]).is_file()


def test_ssm_vae_shapes_and_context_only_rollout() -> None:
    try:
        import torch  # noqa: F401
    except Exception:
        return

    rng = np.random.default_rng(31)
    train = rng.normal(size=(8, 10, 4)).astype(np.float32)
    val = rng.normal(size=(4, 10, 4)).astype(np.float32)
    test = rng.normal(size=(3, 10, 4)).astype(np.float32)
    manifest = ibl.IBLManifest(
        window_seconds=(0.0, 1.0),
        bin_size_seconds=0.1,
        context_bins=4,
        target_bins=6,
        model_suite=["ssm_vae"],
        neural_seeds=[7],
        max_epochs=1,
        patience=1,
        batch_size=4,
        ssm_vae_latent_dim=3,
        ssm_vae_hidden_dim=8,
        ssm_vae_samples=2,
    )

    pred, samples, meta = ibl._train_ssm_vae_forecaster(train, val, test.copy(), manifest, seed=7)

    assert pred.shape == test.shape
    assert samples.shape == (2, *test.shape)
    np.testing.assert_allclose(pred[:, :4, :], test[:, :4, :])
    np.testing.assert_allclose(samples[:, :, :4, :], np.repeat(test[None, :, :4, :], 2, axis=0))
    assert meta["rollout_mode"] == "context_only_latent_prior_sampling"
    pooled_gt, pooled_pred = ibl._pool_stochastic_samples(test, samples)
    assert pooled_gt.shape == pooled_pred.shape == (6, 10, 4)
