from __future__ import annotations

import argparse
import logging
from pathlib import Path

from nethobench.neuro.pipeline import compute_neuro_scores, run_neuro_full_analysis
from nethobench.neuro.fidelity import compute_fidelity_scores
from nethobench.cli.utils import (
    _prompt_for_file,
    _quiet_call,
    _print_scores,
    _save_json_payload,
)

logger = logging.getLogger(__name__)


def _run_neuro(args: argparse.Namespace) -> None:
    gt = _prompt_for_file("ground-truth", "gt_", args.gt)
    preds = _prompt_for_file("inference", "inference_", args.preds)
    scores = _quiet_call(compute_neuro_scores, preds, gt)
    _print_scores("Neuro scores", scores)
    out = _save_json_payload(
        {"scores": scores}, requested=args.json_out, command="neuro-scores", preds=preds
    )
    logger.info(f"Saved scores to {out}")


def _run_fidelity(args: argparse.Namespace) -> None:
    gt = _prompt_for_file("ground-truth", "gt_", args.gt)
    preds = _prompt_for_file("inference", "inference_", args.preds)
    scores = _quiet_call(compute_fidelity_scores, preds, gt)
    _print_scores("Fidelity scores", scores)
    out = _save_json_payload(
        {"scores": scores},
        requested=args.json_out,
        command="fidelity-scores",
        preds=preds,
    )
    logger.info(f"Saved scores to {out}")


def _run_neuro_full(args: argparse.Namespace) -> None:
    gt = _prompt_for_file("ground-truth", "gt_", args.gt)
    preds = _prompt_for_file("inference", "inference_", args.preds)
    out = _quiet_call(run_neuro_full_analysis, preds, gt, output_root=args.output_root)


def add_neuro_subparsers(subparsers) -> None:
    neuro = subparsers.add_parser("neuro-scores", help="Compute neural-only scores.")
    neuro.add_argument(
        "--gt", help="Ground-truth neural CSV (auto-detected if omitted)."
    )
    neuro.add_argument(
        "--preds", help="Predicted neural CSV (auto-detected if omitted)."
    )
    neuro.add_argument("--json-out", help="Optional JSON output path.")
    neuro.set_defaults(func=_run_neuro)

    fidelity = subparsers.add_parser(
        "fidelity-scores", help="Compute the separate fidelity-only neural family."
    )
    fidelity.add_argument(
        "--gt", help="Ground-truth neural CSV (auto-detected if omitted)."
    )
    fidelity.add_argument(
        "--preds", help="Predicted neural CSV (auto-detected if omitted)."
    )
    fidelity.add_argument("--json-out", help="Optional JSON output path.")
    fidelity.set_defaults(func=_run_fidelity)

    neuro_full = subparsers.add_parser(
        "neuro-analysis",
        help="Run full NeuroBench analysis and save figures.",
    )
    neuro_full.add_argument(
        "--gt", help="Ground-truth neural CSV (auto-detected if omitted)."
    )
    neuro_full.add_argument(
        "--preds", help="Predicted neural CSV (auto-detected if omitted)."
    )
    neuro_full.add_argument(
        "--output-root", type=Path, help="Output root (default ./outputs/)."
    )
    neuro_full.set_defaults(func=_run_neuro_full)
