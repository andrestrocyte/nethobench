from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np

from nethobench.cross.pipeline import compute_cross_scores, run_cross_full_analysis
from nethobench.cli.utils import (
    prompt_for_file,
    prompt_for_config,
    quiet_call,
    print_scores,
    print_composite,
)

logger = logging.getLogger(__name__)


def _run_cross(args: argparse.Namespace) -> None:
    gt = prompt_for_file("ground-truth", "gt_", args.gt)
    preds = prompt_for_file("inference", "inference_", args.preds)
    cfg_path = prompt_for_config(args.config)
    scores = quiet_call(compute_cross_scores, preds, gt, cfg_path)
    print_scores("Neuro scores", scores["neuro_scores"])
    print_scores("Behavior scores", scores["behavior_scores"])
    print_scores("Cross-modal scores", scores["cross_scores"])
    print_composite("Composite neuro", scores.get("neuro_composite", float("nan")))
    print_composite("Composite etho", scores.get("etho_composite", float("nan")))
    print_composite("Composite cross", scores.get("cross_composite", float("nan")))
    print_composite("Final composite", scores.get("composite", float("nan")))

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
    logger.info(f"Saved scores to {out}")


def _run_cross_full(args: argparse.Namespace) -> None:
    gt = prompt_for_file("ground-truth", "gt_", args.gt)
    preds = prompt_for_file("inference", "inference_", args.preds)
    cfg_path = prompt_for_config(args.config)
    outdir = quiet_call(
        run_cross_full_analysis, preds, gt, cfg_path, output_root=args.output_root
    )
    logger.info(f"Cross-modal full analysis executed. Outputs under {outdir}")


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
