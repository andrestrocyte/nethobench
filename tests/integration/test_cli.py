"""Integration tests for the nethobench CLI commands.

These tests mirror the commands used in scripts/calculate_scores_and_compare.py
but exercise the CLI against the small fixture datasets under
 tests/resources/data/.
"""
from __future__ import annotations

import json
import shutil
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

# Custom-schema fixtures for config argument testing
_CUSTOM_NEURO_GT = _TEST_DATA / "neural_custom" / "gt-custom.csv"
_CUSTOM_NEURO_PREDS = _TEST_DATA / "neural_custom" / "preds-custom.csv"
_CUSTOM_NEURO_CONFIG = _TEST_DATA / "neural_custom" / "custom_config.json"

_CUSTOM_CROSS_GT = _TEST_DATA / "cross_custom" / "gt-custom.csv"
_CUSTOM_CROSS_PREDS = _TEST_DATA / "cross_custom" / "preds-custom.csv"
_CUSTOM_CROSS_CONFIG = _TEST_DATA / "cross_custom" / "custom_config.json"


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a CLI command and return the completed process."""
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)


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


# ---------------------------------------------------------------------------
# Config argument integration tests
# ---------------------------------------------------------------------------

# Scenario 1: The "Happy Path" (Explicit Config)
@pytest.mark.parametrize("command", ["neuro-scores", "fidelity-scores"])
def test_cli_explicit_config_happy_path(command: str, tmp_path: Path):
    """A valid --config should allow the CLI to parse non-standard CSVs."""
    json_out = tmp_path / "scores.json"
    cmd = [
        "nethobench",
        command,
        "--gt",
        str(_CUSTOM_NEURO_GT),
        "--preds",
        str(_CUSTOM_NEURO_PREDS),
        "--config",
        str(_CUSTOM_NEURO_CONFIG),
        "--json-out",
        str(json_out),
    ]
    result = _run(cmd)
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert json_out.exists()
    payload = json.loads(json_out.read_text())
    assert "scores" in payload


# Scenario 2: Semantic Enforcement (Absence of Config)
@pytest.mark.parametrize("command", ["neuro-scores", "fidelity-scores"])
def test_cli_without_config_fails_on_custom_schema(command: str, tmp_path: Path):
    """Omitting --config on custom-schema CSVs should yield a schema validation error."""
    json_out = tmp_path / "scores.json"
    cmd = [
        "nethobench",
        command,
        "--gt",
        str(_CUSTOM_NEURO_GT),
        "--preds",
        str(_CUSTOM_NEURO_PREDS),
        "--json-out",
        str(json_out),
    ]
    result = _run(cmd)
    assert result.returncode != 0
    stderr = result.stderr
    assert (
        "DataValidationError" in stderr or "Missing required alignment columns" in stderr
    ), f"Expected schema validation error, got: {stderr}"


# Scenario 3: File System Errors (Missing Config)
@pytest.mark.parametrize(
    "command,gt_arg,preds_arg,gt_flag,preds_flag",
    [
        (
            "neuro-scores",
            str(_CUSTOM_NEURO_GT),
            str(_CUSTOM_NEURO_PREDS),
            "--gt",
            "--preds",
        ),
        (
            "etho-scores",
            str(_TEST_DATA / "behavioural" / "behav-ground-truth.csv"),
            str(_TEST_DATA / "behavioural" / "behav-predictions.csv"),
            "--gt-dir",
            "--inf-dir",
        ),
        (
            "cross-scores",
            str(_TEST_DATA / "cross" / "cross-ground-truth.csv"),
            str(_TEST_DATA / "cross" / "cross-predictions.csv"),
            "--gt",
            "--preds",
        ),
    ],
)
def test_cli_missing_config_file(
    command: str,
    gt_arg: str,
    preds_arg: str,
    gt_flag: str,
    preds_flag: str,
    tmp_path: Path,
):
    """A non-existent --config should fail fast with FileNotFoundError."""
    json_out = tmp_path / "scores.json"
    cmd = [
        "nethobench",
        command,
        gt_flag,
        gt_arg,
        preds_flag,
        preds_arg,
        "--config",
        "does_not_exist.json",
        "--json-out",
        str(json_out),
    ]
    result = _run(cmd)
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr


# Scenario 4: Malformed Config File
MALFORMED_CONFIGS = [
    pytest.param('{"invalid": json,}', id="trailing_comma"),
    pytest.param("not json at all", id="raw_text"),
    pytest.param('"just a string"', id="json_string_not_object"),
]


@pytest.mark.parametrize("malformed_content", MALFORMED_CONFIGS)
@pytest.mark.parametrize(
    "command,gt_arg,preds_arg,gt_flag,preds_flag",
    [
        (
            "neuro-scores",
            str(_CUSTOM_NEURO_GT),
            str(_CUSTOM_NEURO_PREDS),
            "--gt",
            "--preds",
        ),
        (
            "etho-scores",
            str(_TEST_DATA / "behavioural" / "behav-ground-truth.csv"),
            str(_TEST_DATA / "behavioural" / "behav-predictions.csv"),
            "--gt-dir",
            "--inf-dir",
        ),
        (
            "cross-scores",
            str(_TEST_DATA / "cross" / "cross-ground-truth.csv"),
            str(_TEST_DATA / "cross" / "cross-predictions.csv"),
            "--gt",
            "--preds",
        ),
    ],
)
def test_cli_malformed_config_file(
    command: str,
    gt_arg: str,
    preds_arg: str,
    gt_flag: str,
    preds_flag: str,
    malformed_content: str,
    tmp_path: Path,
):
    """A malformed --config should yield a JSON decoding or schema error."""
    malformed = tmp_path / "malformed.json"
    malformed.write_text(malformed_content)
    json_out = tmp_path / "scores.json"
    cmd = [
        "nethobench",
        command,
        gt_flag,
        gt_arg,
        preds_flag,
        preds_arg,
        "--config",
        str(malformed),
        "--json-out",
        str(json_out),
    ]
    result = _run(cmd)
    assert result.returncode != 0
    stderr = result.stderr
    assert (
        "JSONDecodeError" in stderr
        or "Config file must contain a JSON object" in stderr
        or "Expecting property name" in stderr
        or "Expecting value" in stderr
    ), f"Expected JSON error, got: {stderr}"


# Scenario 5: Auto-Discovery (Single JSON in Directory)
def test_cli_auto_discovery_single_json(tmp_path: Path):
    """With exactly one JSON in cwd, the CLI should silently auto-infer the config."""
    shutil.copy(_CUSTOM_NEURO_GT, tmp_path / "gt.csv")
    shutil.copy(_CUSTOM_NEURO_PREDS, tmp_path / "preds.csv")
    shutil.copy(_CUSTOM_NEURO_CONFIG, tmp_path / "custom_config.json")

    json_out = tmp_path / "scores.json"
    cmd = [
        "nethobench",
        "neuro-scores",
        "--gt",
        str(tmp_path / "gt.csv"),
        "--preds",
        str(tmp_path / "preds.csv"),
        "--json-out",
        str(json_out),
    ]
    result = _run(cmd, cwd=tmp_path)
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert json_out.exists()
    payload = json.loads(json_out.read_text())
    assert "scores" in payload
    # Success implies the config was auto-discovered, because custom-schema CSVs
    # would fail with the default schema (see test_cli_without_config_fails_on_custom_schema).


# Scenario 6: Ambiguous Auto-Discovery (Non-Interactive Failure)
def test_cli_auto_discovery_ambiguous_json(tmp_path: Path):
    """With multiple JSONs in cwd and no TTY, the CLI should fail with EOFError."""
    shutil.copy(_CUSTOM_NEURO_GT, tmp_path / "gt.csv")
    shutil.copy(_CUSTOM_NEURO_PREDS, tmp_path / "preds.csv")
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "b.json").write_text("{}")

    json_out = tmp_path / "scores.json"
    cmd = [
        "nethobench",
        "neuro-scores",
        "--gt",
        str(tmp_path / "gt.csv"),
        "--preds",
        str(tmp_path / "preds.csv"),
        "--json-out",
        str(json_out),
    ]
    # Close stdin by piping from /dev/null equivalent (subprocess with no input)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode != 0
    assert "EOFError" in result.stderr
