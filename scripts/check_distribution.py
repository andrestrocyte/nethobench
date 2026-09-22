"""Verify wheel contents and smoke-test the installed package outside the repo."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--python", required=True, type=Path)
    args = parser.parse_args()
    with ZipFile(args.wheel) as archive:
        names = archive.namelist()
        assert not any(n.startswith(("tests/", "data/", "outputs/", "paper/", "scripts/")) for n in names)
        for name in ("nethobench/neuro/metrics/temporal.py", "nethobench/probabilistic.py",
                     "nethobench/diagnostics/implementation_sensitivity.py"):
            assert name in names, name
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory() as cwd:
        probe = subprocess.run([str(args.python.absolute()), "-c",
            "import nethobench; print(nethobench.__file__)"],
            cwd=cwd, env=env, check=True, capture_output=True, text=True)
        assert "site-packages" in probe.stdout, probe.stdout
        subprocess.run([str(args.python.absolute()), "-m", "nethobench.cli.main", "--help"],
                       cwd=cwd, env=env, check=True, capture_output=True)
    print(json.dumps({"wheel_entries": len(names), "installed_import": probe.stdout.strip(),
                      "outside_repo_cli": "passed"}))


if __name__ == "__main__":
    main()
