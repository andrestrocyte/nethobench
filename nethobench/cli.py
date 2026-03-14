from __future__ import annotations

import argparse
import contextlib
from datetime import datetime
import io
import json
from pathlib import Path
from typing import Iterable, Optional

from nethobench import (
    compute_cross_scores,
    compute_etho_scores,
    compute_neuro_scores,
    run_cross_full_analysis,
    run_ethobench_notebook,
    run_neuro_full_analysis,
)


def _find_candidates(prefix: str) -> list[Path]:
    cwd = Path.cwd()
    return sorted(path for path in cwd.glob(f"{prefix}*") if path.is_file())


def _prompt_for_file(label: str, prefix: str, provided: Optional[str]) -> Path:
    if provided:
        path = Path(provided).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file():
            raise FileNotFoundError(f"{label} file '{path}' does not exist.")
        return path

    candidates = _find_candidates(prefix)
    if candidates:
        print(f"Detected {label.lower()} candidates in {Path.cwd()}:")
        for idx, candidate in enumerate(candidates, start=1):
            print(f"  [{idx}] {candidate.name}")

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
                print("Invalid selection number. Try again.")
                continue
        elif response:
            selection = Path(response)
        else:
            print("Please provide a filename or choose one of the listed entries.")
            continue

        selection = selection.expanduser()
        if not selection.is_absolute():
            selection = Path.cwd() / selection
        if selection.is_file():
            return selection
        print(f"{selection} does not exist. Try again.")


def _prompt_for_config(provided: Optional[str]) -> Optional[Path]:
    if provided:
        return Path(provided)
    jsons = sorted(Path.cwd().glob("*.json"))
    if len(jsons) == 1:
        return jsons[0]
    if jsons:
        print("Detected possible config JSON files:")
        for idx, candidate in enumerate(jsons, start=1):
            print(f"  [{idx}] {candidate.name}")
        response = input("Enter config filename (or leave blank to auto-infer): ").strip()
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


def _score_to_color(value: float) -> str:
    try:
        v = float(value)
    except Exception:
        return ""
    if v >= 0.8:
        return "\033[32m"
    if v >= 0.4:
        return "\033[33m"
    return "\033[31m"


def _render_score_bar(value: float, width: int = 16) -> str:
    try:
        v = float(value)
    except Exception:
        return ""
    v = max(0.0, min(1.0, v))
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
    color = _score_to_color(v)
    reset = "\033[0m"
    return f"{color}{icon} 0 {''.join(chars)} 1{reset}"


def _print_scores(label: str, scores: dict[str, float]) -> None:
    print(f"\n{label}:")
    for key, value in scores.items():
        if value == value:
            print(f"  {key:24s}: {value:.3f} {_render_score_bar(value)}")
        else:
            print(f"  {key:24s}: NaN")


def _print_composite(label: str, value: float) -> None:
    if value == value:
        print(f"{label:18s}: {value:.3f} {_render_score_bar(value)}")
    else:
        print(f"{label:18s}: NaN")


def _quiet_call(func, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        return func(*args, **kwargs)


def _default_json_output(command: str, preds: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = Path.cwd() / "outputs" / f"{preds.stem}-{command}-{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / "scores.json"


def _save_json_payload(payload: dict, *, requested: Optional[str], command: str, preds: Path) -> Path:
    out = Path(requested) if requested else _default_json_output(command, preds)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nethobench",
        description="Unified neural + behavioral + cross-modal benchmarking.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    neuro = subparsers.add_parser("neuro-scores", help="Compute neural-only scores.")
    neuro.add_argument("--gt", help="Ground-truth neural CSV (auto-detected if omitted).")
    neuro.add_argument("--preds", help="Predicted neural CSV (auto-detected if omitted).")
    neuro.add_argument("--json-out", help="Optional JSON output path.")
    neuro.set_defaults(func=_run_neuro)

    neuro_full = subparsers.add_parser(
        "neuro-analysis",
        help="Run full NeuroBench analysis and save figures.",
    )
    neuro_full.add_argument("--gt", help="Ground-truth neural CSV (auto-detected if omitted).")
    neuro_full.add_argument("--preds", help="Predicted neural CSV (auto-detected if omitted).")
    neuro_full.add_argument("--ddconfig", required=True, help="ddconfig JSON used by the notebook.")
    neuro_full.add_argument("--output-root", type=Path, help="Output root (default ./outputs/).")
    neuro_full.set_defaults(func=_run_neuro_full)

    etho = subparsers.add_parser("etho-scores", help="Compute behavior-only scores.")
    etho.add_argument("--gt-dir", required=True, help="Directory with GT pose parquet/CSV.")
    etho.add_argument("--inf-dir", required=True, help="Directory with inference pose parquet/CSV.")
    etho.add_argument("--json-out", help="Optional JSON output path.")
    etho.add_argument("--run-notebook", action="store_true", help="Also execute the bundled ethobench notebook.")
    etho.add_argument("--output-root", type=Path, help="Output root for notebook capture.")
    etho.set_defaults(func=_run_etho)

    cross = subparsers.add_parser(
        "cross-scores",
        help="Compute neuro + behavior + cross-modal scores from multimodal CSVs.",
    )
    cross.add_argument("--gt", help="Ground-truth multimodal CSV (auto-detected if omitted).")
    cross.add_argument("--preds", help="Predicted multimodal CSV (auto-detected if omitted).")
    cross.add_argument("--config", help="JSON config describing neuro/behavior columns (auto-inferred if omitted).")
    cross.add_argument("--json-out", help="Optional JSON output path.")
    cross.set_defaults(func=_run_cross)

    cross_full = subparsers.add_parser(
        "cross-analysis",
        help="Execute cross-modal notebook headlessly and save figures + executed notebook.",
    )
    cross_full.add_argument("--gt", help="Ground-truth multimodal CSV (auto-detected if omitted).")
    cross_full.add_argument("--preds", help="Predicted multimodal CSV (auto-detected if omitted).")
    cross_full.add_argument("--config", help="JSON config describing neuro/behavior columns (auto-inferred if omitted).")
    cross_full.add_argument("--output-root", type=Path, help="Output root (default ./outputs/).")
    cross_full.set_defaults(func=_run_cross_full)

    return parser


def _run_neuro(args: argparse.Namespace) -> None:
    gt = _prompt_for_file("ground-truth", "gt_", args.gt)
    preds = _prompt_for_file("inference", "inference_", args.preds)
    scores = _quiet_call(compute_neuro_scores, preds, gt)
    _print_scores("Neuro scores", scores)
    out = _save_json_payload({"scores": scores}, requested=args.json_out, command="neuro-scores", preds=preds)
    print(f"Saved scores to {out}")


def _run_neuro_full(args: argparse.Namespace) -> None:
    gt = _prompt_for_file("ground-truth", "gt_", args.gt)
    preds = _prompt_for_file("inference", "inference_", args.preds)
    out = _quiet_call(run_neuro_full_analysis, preds, gt, Path(args.ddconfig), output_root=args.output_root)
    _print_scores("Neuro analysis scores", out["scores"])
    print(f"\nSaved scores to {out['scores_path']}")
    print(f"Wrapped notebook saved to {out['wrapped_notebook']}")
    print(f"Plots and outputs under {out['output_dir']}")


def _run_etho(args: argparse.Namespace) -> None:
    scores, seq_scores, seq_means, seq_stds = _quiet_call(compute_etho_scores, Path(args.gt_dir), Path(args.inf_dir))
    _print_scores("Behavior scores", scores)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "global_scores": scores,
            "per_sequence": seq_scores,
            "per_sequence_mean": seq_means,
            "per_sequence_std": seq_stds,
        }
        out.write_text(json.dumps(payload, indent=2))
        print(f"Saved scores to {out}")
    if args.run_notebook:
        outdir = run_ethobench_notebook(Path(args.gt_dir), Path(args.inf_dir), output_root=args.output_root)
        print(f"Notebook executed. Outputs in {outdir}")


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
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(scores, indent=2))
        print(f"Saved scores to {out}")


def _run_cross_full(args: argparse.Namespace) -> None:
    gt = _prompt_for_file("ground-truth", "gt_", args.gt)
    preds = _prompt_for_file("inference", "inference_", args.preds)
    cfg_path = _prompt_for_config(args.config)
    outdir = _quiet_call(run_cross_full_analysis, preds, gt, cfg_path, output_root=args.output_root)
    print(f"Cross-modal full analysis executed. Outputs under {outdir}")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        parser.exit(1, "\nCancelled.\n")


if __name__ == "__main__":  # pragma: no cover
    main()
