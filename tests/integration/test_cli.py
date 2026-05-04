"""Integration tests for the nethobench CLI commands.

These tests mirror the commands used in scripts/calculate_scores_and_compare.py
but exercise the CLI against the small fixture datasets under
 tests/resources/data/.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_TEST_DATA = _PROJECT_ROOT / "tests" / "resources" / "data"

NEURAL_FIXTURES = [
    (
        _TEST_DATA / "neural" / "ground-truth" / "gt-ba4.csv",
        _TEST_DATA / "neural" / "predictions" / "preds-ba4.csv",
    ),
    (
        _TEST_DATA / "neural" / "ground-truth" / "gt-ba16.csv",
        _TEST_DATA / "neural" / "predictions" / "preds-ba16.csv",
    ),
]

ETHO_FIXTURES = [
    (
        _TEST_DATA / "behavioural" / "behav-ground-truth.csv",
        _TEST_DATA / "behavioural" / "behav-predictions.csv",
    ),
]

CROSS_FIXTURES = [
    (
        _TEST_DATA / "cross" / "cross-ground-truth.csv",
        _TEST_DATA / "cross" / "cross-predictions.csv",
    ),
]


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a CLI command and return the completed process."""
    return subprocess.run(cmd, capture_output=True, text=True)


@pytest.mark.parametrize("gt,preds", NEURAL_FIXTURES)
def test_cli_neuro_scores(gt: Path, preds: Path, tmp_path: Path):
    json_out = tmp_path / "neuro-scores.json"
    cmd = [
        "nethobench",
        "neuro-scores",
        "--gt",
        str(gt),
        "--preds",
        str(preds),
        "--json-out",
        str(json_out),
    ]
    result = _run(cmd)
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert json_out.exists()
    payload = json.loads(json_out.read_text())
    assert "scores" in payload
    assert "composite_score" in payload["scores"]


@pytest.mark.parametrize("gt,preds", NEURAL_FIXTURES)
def test_cli_fidelity_scores(gt: Path, preds: Path, tmp_path: Path):
    json_out = tmp_path / "fidelity-scores.json"
    cmd = [
        "nethobench",
        "fidelity-scores",
        "--gt",
        str(gt),
        "--preds",
        str(preds),
        "--json-out",
        str(json_out),
    ]
    result = _run(cmd)
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert json_out.exists()
    payload = json.loads(json_out.read_text())
    assert "scores" in payload


@pytest.mark.parametrize("gt,preds", NEURAL_FIXTURES)
def test_cli_neuro_analysis(gt: Path, preds: Path, tmp_path: Path):
    output_root = tmp_path / "neuro-analysis"
    cmd = [
        "nethobench",
        "neuro-analysis",
        "--gt",
        str(gt),
        "--preds",
        str(preds),
        "--output-root",
        str(output_root),
    ]
    result = _run(cmd)
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    # The command creates a sub-directory named after the predictions stem.
    subdirs = [d for d in output_root.iterdir() if d.is_dir()]
    assert subdirs, "Expected at least one output subdirectory"
    scores_json = subdirs[0] / "scores.json"
    assert scores_json.exists()
    payload = json.loads(scores_json.read_text())
    assert "composite_score" in payload


@pytest.mark.parametrize("gt,preds", ETHO_FIXTURES)
def test_cli_etho_scores(gt: Path, preds: Path, tmp_path: Path):
    json_out = tmp_path / "etho-scores"
    cmd = [
        "nethobench",
        "etho-scores",
        "--gt-dir",
        str(gt),
        "--inf-dir",
        str(preds),
        "--json-out",
        str(json_out),
    ]
    result = _run(cmd)
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    scores_json = json_out / "scores.json"
    assert scores_json.exists()
    payload = json.loads(scores_json.read_text())
    assert "global_scores" in payload
    assert "per_sequence" in payload
    assert "per_sequence_mean" in payload
    assert "per_sequence_std" in payload


@pytest.mark.parametrize("gt,preds", CROSS_FIXTURES)
def test_cli_cross_scores(gt: Path, preds: Path, tmp_path: Path):
    json_out = tmp_path / "cross-scores"
    cmd = [
        "nethobench",
        "cross-scores",
        "--gt",
        str(gt),
        "--preds",
        str(preds),
        "--json-out",
        str(json_out),
    ]
    result = _run(cmd)
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    scores_json = json_out / "scores.json"
    assert scores_json.exists()
    payload = json.loads(scores_json.read_text())
    assert "neuro_scores" in payload
    assert "behavior_scores" in payload
    assert "cross_scores" in payload
    assert "composite" in payload

@pytest.mark.parametrize("gt,preds", ETHO_FIXTURES)
def test_cli_etho_analysis(gt: Path, preds: Path, tmp_path: Path):
    output_root = tmp_path / "etho-analysis"
    cmd = [
        "nethobench",
        "etho-analysis",
        "--gt-dir",
        str(gt),
        "--inf-dir",
        str(preds),
        "--output-root",
        str(output_root),
    ]
    result = _run(cmd)
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    
    # The command creates a sub-directory named with a timestamp prefix
    subdirs = [d for d in output_root.iterdir() if d.is_dir()]
    assert subdirs, "Expected at least one output subdirectory"
    
    scores_json = subdirs[0] / "scores.json"
    assert scores_json.exists()
    
    payload = json.loads(scores_json.read_text())
    assert "global_scores" in payload
    assert "per_sequence" in payload


@pytest.mark.parametrize("gt,preds", CROSS_FIXTURES)
def test_cli_cross_analysis(gt: Path, preds: Path, tmp_path: Path):
    output_root = tmp_path / "cross-analysis"
    cmd = [
        "nethobench",
        "cross-analysis",
        "--gt",
        str(gt),
        "--preds",
        str(preds),
        "--output-root",
        str(output_root),
    ]
    result = _run(cmd)
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    
    # The command creates a sub-directory named after the predictions stem
    subdirs = [d for d in output_root.iterdir() if d.is_dir()]
    assert subdirs, "Expected at least one output subdirectory"
    
    scores_json = subdirs[0] / "scores.json"
    assert scores_json.exists()
    
    payload = json.loads(scores_json.read_text())
    assert "cross_scores" in payload
    assert "cross_composite" in payload
    assert "composite" in payload