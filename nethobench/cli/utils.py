from __future__ import annotations

import logging
import os
import json
from pathlib import Path
from typing import Optional

import numpy as np

from nethobench.utils.helpers import quiet_call

logger = logging.getLogger(__name__)


def find_candidates(prefix: str) -> list[Path]:
    cwd = Path.cwd()
    return sorted(path for path in cwd.glob(f"{prefix}*") if path.is_file())


def prompt_for_file(label: str, prefix: str, provided: Optional[str]) -> Path:
    if provided:
        path = Path(provided).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file():
            raise FileNotFoundError(f"{label} file '{path}' does not exist.")
        return path

    candidates = find_candidates(prefix)
    if candidates:
        logger.info(f"Detected {label.lower()} candidates in {Path.cwd()}:")
        for idx, candidate in enumerate(candidates, start=1):
            logger.info(f"  [{idx}] {candidate.name}")

    default_candidate = candidates[0] if len(candidates) == 1 else None

    while True:
        prompt = f"Enter {label} filename"
        if default_candidate is not None:
            prompt += f" [{default_candidate.name}]"
        prompt += ": "

        response = input(prompt).strip()
        if not response and default_candidate is not None:
            selection = default_candidate
        elif response.isdigit() and candidates:
            idx = int(response) - 1
            if 0 <= idx < len(candidates):
                selection = candidates[idx]
            else:
                logger.warning("Invalid selection number. Try again.")
                continue
        elif response:
            selection = Path(response)
        else:
            logger.warning("Please provide a filename or choose one of the listed entries.")
            continue

        selection = selection.expanduser()
        if not selection.is_absolute():
            selection = Path.cwd() / selection
        if selection.is_file():
            return selection
        logger.warning(f"{selection} does not exist. Try again.")


def prompt_for_config(provided: Optional[str]) -> Optional[Path]:
    if provided:
        return Path(provided)
    jsons = sorted(Path.cwd().glob("*.json"))
    if len(jsons) == 1:
        return jsons[0]
    if jsons:
        logger.info("Detected possible config JSON files:")
        for idx, candidate in enumerate(jsons, start=1):
            logger.info(f"  [{idx}] {candidate.name}")
        response = input(
            "Enter config filename (or leave blank to auto-infer): "
        ).strip()
        if response.isdigit():
            idx = int(response) - 1
            if 0 <= idx < len(jsons):
                return jsons[idx]
        elif response:
            path = Path(response)
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.is_file():
                return path
    return None


def score_to_color(value: float) -> str:
    v = float(value)
    if v >= 0.8:
        return "\033[32m"
    if v >= 0.4:
        return "\033[33m"
    return "\033[31m"


def render_score_bar(value: float, width: int = 16) -> str:
    v = max(0.0, min(1.0, float(value)))
    arrow_idx = min(width - 1, max(0, int(round(v * (width - 1)))))
    icon = "↗" if v >= 0.8 else ("→" if v >= 0.4 else "↘")
    chars = []
    for idx in range(width):
        if idx < arrow_idx:
            chars.append("━")
        elif idx == arrow_idx:
            chars.append("▶")
        else:
            chars.append("─")
    color = score_to_color(v)
    reset = "\033[0m"
    return f"{color}{icon} 0 {''.join(chars)} 1{reset}"


def print_scores(label: str, scores: dict[str, float]) -> None:
    logger.info(f"\n{label}:")
    for key, value in scores.items():
        logger.info(f"  {key:24s}: {value:.3f} {render_score_bar(value)}")


def print_composite(label: str, value: float) -> None:
    if value == value:
        logger.info(f"{label:18s}: {value:.3f} {render_score_bar(value)}")
    else:
        logger.info(f"{label:18s}: NaN")


def default_json_output(command: str, preds: Path) -> Path:
    outdir = Path.cwd() / "outputs" / f"{preds.stem}-{command}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / "scores.json"


def default_output_dir(command: str) -> Path:
    outdir = Path.cwd() / "outputs" / f"{command}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def save_json_payload(
    payload: dict, *, requested: Optional[str], command: str, preds: Path
) -> Path:
    out = Path(requested) if requested else default_json_output(command, preds)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    return out
