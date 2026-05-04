from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np

from nethobench.etho.pipeline import compute_etho_scores, run_etho_full_analysis
from nethobench.cli.utils import prompt_for_file, quiet_call, print_scores

logger = logging.getLogger(__name__)


def _run_etho(args: argparse.Namespace) -> None:
    scores, seq_scores, seq_means, seq_stds = quiet_call(
        compute_etho_scores, Path(args.gt_dir), Path(args.inf_dir)
    )
    print_scores("Behavior scores", scores)
    if args.json_out:
        out = Path(args.json_out)
    else:
        out = Path(
            os.path.join(
                "outputs", f"{args.inf_dir.split(os.sep)[-1].split('.')[0]}-etho-scores"
            )
        )

    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "global_scores": scores,
        "per_sequence": seq_scores,
        "per_sequence_mean": seq_means,
        "per_sequence_std": seq_stds,
    }
    with open(os.path.join(out, "scores.json"), "w") as f:
        f.write(json.dumps(payload, indent=2))

    logger.info(f"Saved scores to {out}")


def _run_etho_full(args: argparse.Namespace) -> None:
    outdir = quiet_call(
        run_etho_full_analysis,
        Path(args.gt_dir),
        Path(args.inf_dir),
        output_root=args.output_root,
    )
    logger.info(f"Behavioral full analysis executed natively. Outputs under {outdir}")


def add_etho_subparsers(subparsers) -> None:
    etho = subparsers.add_parser("etho-scores", help="Compute behavior-only scores.")
    etho.add_argument(
        "--gt-dir", required=True, help="Directory with GT pose parquet/CSV."
    )
    etho.add_argument(
        "--inf-dir", required=True, help="Directory with inference pose parquet/CSV."
    )
    etho.add_argument("--json-out", help="Optional JSON output path.")
    etho.set_defaults(func=_run_etho)

    etho_full = subparsers.add_parser(
        "etho-analysis",
        help="Run full EthoBench analysis and save figures natively.",
    )
    etho_full.add_argument(
        "--gt-dir", required=True, help="Directory with GT pose parquet/CSV."
    )
    etho_full.add_argument(
        "--inf-dir", required=True, help="Directory with inference pose parquet/CSV."
    )
    etho_full.add_argument(
        "--output-root", type=Path, help="Output root (default ./outputs/)."
    )
    etho_full.set_defaults(func=_run_etho_full)
