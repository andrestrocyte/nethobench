"""Compare installed-wheel CLI scores with the paper's source in one environment.

No golden numbers are regenerated from the candidate. The reference is exported
from the immutable paper commit. Run after installing the candidate wheel.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

BASE_COMMIT = "4075d2fe13b354de910d0cd1bb22826b94296594"


def numeric_leaves(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(numeric_leaves(child, prefix + "/" + key))
        return result
    if isinstance(value, list):
        result = {}
        for i, child in enumerate(value):
            result.update(numeric_leaves(child, prefix + "/" + str(i)))
        return result
    if isinstance(value, (int, float)) or value is None:
        return {prefix: value}
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fixtures = root / "tests/resources/data"
    cases = []
    for suffix in ("ba4", "ba16"):
        for command in ("neuro-scores", "fidelity-scores"):
            cases.append((f"{command}-{suffix}", command, "--gt", "--preds",
                          fixtures / f"neural/ground-truth/gt-{suffix}.csv",
                          fixtures / f"neural/predictions/preds-{suffix}.csv", False))
    cases += [
        ("etho-scores", "etho-scores", "--gt-dir", "--inf-dir",
         fixtures / "behavioural/behav-ground-truth.csv",
         fixtures / "behavioural/behav-predictions.csv", True),
        ("cross-scores", "cross-scores", "--gt", "--preds",
         fixtures / "cross/cross-ground-truth.csv",
         fixtures / "cross/cross-predictions.csv", True),
    ]
    report = {"baseline_commit": BASE_COMMIT, "relative_tolerance": 1e-10,
              "absolute_tolerance": 1e-12, "cases": []}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        archive = tmp / "baseline.tar"
        with archive.open("wb") as output:
            subprocess.run(["git", "archive", BASE_COMMIT, "nethobench"], cwd=root,
                           stdout=output, check=True)
        baseline = (tmp / "baseline").resolve()
        baseline.mkdir()
        # Local trusted git tree; reject paths outside the export directory.
        with tarfile.open(archive) as contents:
            for member in contents.getmembers():
                if not member.isfile() and not member.isdir():
                    raise ValueError("Unexpected non-file in baseline archive")
                if not (baseline / member.name).resolve().is_relative_to(baseline):
                    raise ValueError("Unsafe archive path")
            contents.extractall(baseline)
        for name, command, gtflag, predflag, gt, pred, directory in cases:
            values = []
            for source in ("baseline", "wheel"):
                env = os.environ.copy()
                env.pop("PYTHONPATH", None)
                env.update({"MPLBACKEND": "Agg", "NUMBA_NUM_THREADS": "1",
                            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                            "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"})
                if source == "baseline":
                    env["PYTHONPATH"] = str(baseline)
                output = tmp / f"{name}-{source}"
                subprocess.run([str(args.python.absolute()), "-m", "nethobench.cli.main",
                                command, gtflag, str(gt), predflag, str(pred),
                                "--json-out", str(output)], cwd=tmp, env=env, check=True,
                               capture_output=True, text=True)
                payload = json.loads((output / "scores.json" if directory else output).read_text())
                values.append(numeric_leaves(payload))
            reference, candidate = values
            assert reference.keys() == candidate.keys(), f"Output keys changed: {name}"
            assert reference, f"No numerical outputs: {name}"
            maximum = 0.
            for key, expected in reference.items():
                actual = candidate[key]
                if expected is None or actual is None:
                    assert expected is actual, (name, key, expected, actual)
                elif math.isnan(expected):
                    assert math.isnan(actual), (name, key, expected, actual)
                else:
                    assert math.isclose(expected, actual, rel_tol=1e-10, abs_tol=1e-12), (name, key, expected, actual)
                    if math.isfinite(expected):
                        maximum = max(maximum, abs(expected - actual))
            report["cases"].append({"name": name, "numeric_values": len(reference),
                                    "maximum_absolute_difference": maximum, "passed": True})
            print(f"{name}: {len(reference)} numerical values match", flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
