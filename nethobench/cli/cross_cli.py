from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from nethobench.cross.pipeline import compute_cross_scores, run_cross_full_analysis
from nethobench.cli.utils import (
    _prompt_for_file,
    _prompt_for_config,
    _quiet_call,
    _print_scores,
    _print_composite,
)


def _run_cross(args: argparse.Namespace) -> None:
    gt = _prompt_for_file("ground-truth", "gt_", args.gt)
    preds = _prompt_for_file("inference", "inference_", args.preds)
    cfg_path = _prompt_for_config(args.config)
    scores = _quiet_call(compute_cross_scores, preds, gt, cfg_path)
    _print_scores("Neuro scores", scores["neuro_scores"])
    _print_scores("Behavior scores", scores["behavior_scores"])
    _print_scores("Cross-modal scores", scores["cross_scores"])
    print("")
    _print_composite("Composite neuro", scores.get("neuro_composite", float("nan")))
    _print_composite("Composite etho", scores.get("etho_composite", float("nan")))
    _print_composite("Composite cross", scores.get("cross_composite", float("nan")))
    _print_composite("Final composite", scores.get("composite", float("nan")))

    if args.json_out is not None:
        out = Path(args.json_out)
    else:
        out = Path(
            os.path.join(
                "outputs", f"{args.gt.split(os.sep)[-1].split('.')[0]}-cross-scores"
            )
        )

    out.mkdir(parents=True, exist_ok=True)
    with open(f"{out}/scores.json", "w") as f:
        f.write(json.dumps(scores, indent=2))
    print(f"Saved scores to {out}")


def _run_cross_full(args: argparse.Namespace) -> None:
    gt = _prompt_for_file("ground-truth", "gt_", args.gt)
    preds = _prompt_for_file("inference", "inference_", args.preds)
    cfg_path = _prompt_for_config(args.config)
    outdir = _quiet_call(
        run_cross_full_analysis, preds, gt, cfg_path, output_root=args.output_root
    )
    print(f"Cross-modal full analysis executed. Outputs under {outdir}")


def add_cross_subparsers(subparsers) -> None:
    cross = subparsers.add_parser(
        "cross-scores",
        help="Compute neuro + behavior + cross-modal scores from multimodal CSVs.",
    )
    cross.add_argument(
        "--gt", help="Ground-truth multimodal CSV (auto-detected if omitted)."
    )
    cross.add_argument(
        "--preds", help="Predicted multimodal CSV (auto-detected if omitted)."
    )
    cross.add_argument(
        "--config",
        help="JSON config describing neuro/behavior columns (auto-inferred if omitted).",
    )
    cross.add_argument("--json-out", help="Optional JSON output path.")
    cross.set_defaults(func=_run_cross)

    cross_full = subparsers.add_parser(
        "cross-analysis",
        help="Execute cross-modal notebook headlessly and save figures + executed notebook.",
    )
    cross_full.add_argument(
        "--gt", help="Ground-truth multimodal CSV (auto-detected if omitted)."
    )
    cross_full.add_argument(
        "--preds", help="Predicted multimodal CSV (auto-detected if omitted)."
    )
    cross_full.add_argument(
        "--config",
        help="JSON config describing neuro/behavior columns (auto-inferred if omitted).",
    )
    cross_full.add_argument(
        "--output-root", type=Path, help="Output root (default ./outputs/)."
    )
    cross_full.set_defaults(func=_run_cross_full)
