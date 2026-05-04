from __future__ import annotations

import argparse
import logging
import sys
from typing import Iterable, Optional

from nethobench.cli.neuro_cli import add_neuro_subparsers
from nethobench.cli.etho_cli import add_etho_subparsers
from nethobench.cli.cross_cli import add_cross_subparsers
from nethobench.cli.utils import (
    _find_candidates,
    _prompt_for_file,
    _prompt_for_config,
    _score_to_color,
    _render_score_bar,
    _print_scores,
    _print_composite,
    _default_json_output,
    _default_output_dir,
    _save_json_payload,
)
from nethobench.utils.helpers import _quiet_call


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nethobench",
        description="Unified neural + behavioral + cross-modal benchmarking.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_neuro_subparsers(subparsers)
    add_etho_subparsers(subparsers)
    add_cross_subparsers(subparsers)

    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            stream=sys.stdout,
        )
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        parser.exit(1, "\nCancelled.\n")


if __name__ == "__main__":  # pragma: no cover
    main()
