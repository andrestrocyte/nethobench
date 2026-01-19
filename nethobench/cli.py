from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from nethobench import (
    compute_neuro_scores,
    run_neuro_full_analysis,
    compute_etho_scores,
    compute_cross_scores,
    run_cross_full_analysis,
    run_ethobench_notebook,
)


def _find_candidates(prefix: str) -> list[Path]:
    cwd = Path.cwd()
    return sorted(p for p in cwd.glob(f"{prefix}*") if p.is_file())


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
        resp = input("Enter config filename (or leave blank to auto-infer): ").strip()
        if resp.isdigit() and jsons:
            idx = int(resp) - 1
            if 0 <= idx < len(jsons):
                return jsons[idx]
        elif resp:
            path = Path(resp)
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.is_file():
                return path
    # Allow None to trigger auto-infer
    return None


def _score_to_color(value: float) -> str:
    try:
        v = float(value)
    except Exception:
        return ""
    v = max(0.0, min(1.0, v))
    red = int(round((1.0 - v) * 255))
    green = int(round(v * 255))
    return f"\033[38;2;{red};{green};0m"


def _render_score_bar(value: float, width: int = 24) -> str:
    try:
        v = float(value)
    except Exception:
        return ""
    v = max(0.0, min(1.0, v))
    arrow_idx = min(width - 1, max(0, int(round(v * (width - 1)))))
    chars = []
    for idx in range(width):
        if idx < arrow_idx:
            chars.append("=")
        elif idx == arrow_idx:
            chars.append(">")
        else:
            chars.append(".")
    color = _score_to_color(v)
    reset = "\033[0m"
    return f"{color}0 |{''.join(chars)}| 1{reset}"


def _print_scores(label: str, scores: dict[str, float]) -> None:
    print(f"\n{label}:")
    for k, v in scores.items():
        if v == v:  # NaN check
            bar = _render_score_bar(v)
            print(f"  {k:24s}: {v:.3f} {bar}")
        else:
            print(f"  {k:24s}: NaN")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nethobench",
        description="Unified neural + behavioral + cross-modal benchmarking.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    neuro = subparsers.add_parser("neuro-scores", help="Compute neural-only scores.")
    neuro.add_argument("--gt", help="Ground-truth neural CSV (auto-detected if omitted).")
    neuro.add_argument("--preds", help="Predicted neural CSV (auto-detected if omitted).")
    neuro.add_argument("--per-seq-std", action="store_true", dest="per_seq_std", help="Include per-sequence stats.")
    neuro.add_argument("--json-out", help="Optional JSON output path.")
    neuro.set_defaults(func=_run_neuro)

    neuro_full = subparsers.add_parser(
        "neuro-analysis",
        help="Run full NeuroBench analysis (notebook-derived script) and save figures.",
    )
    neuro_full.add_argument("--gt", help="Ground-truth neural CSV (auto-detected if omitted).")
    neuro_full.add_argument("--preds", help="Predicted neural CSV (auto-detected if omitted).")
    neuro_full.add_argument("--ddconfig", required=True, help="ddconfig JSON used by the original notebook.")
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
    scores = compute_neuro_scores(preds, gt, per_sequence_stats=args.per_seq_std)
    _print_scores("Neuro scores", scores if not args.per_seq_std else scores["pooled_scores"])
    payload = scores if args.per_seq_std else {"pooled_scores": scores}
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"Saved scores to {out}")


def _run_neuro_full(args: argparse.Namespace) -> None:
    gt = _prompt_for_file("ground-truth", "gt_", args.gt)
    preds = _prompt_for_file("inference", "inference_", args.preds)
    out = run_neuro_full_analysis(preds, gt, Path(args.ddconfig), output_root=args.output_root)
    print(f"Neuro full analysis complete. Outputs under {out['output_dir']}")


def _run_etho(args: argparse.Namespace) -> None:
    scores, seq_scores, seq_means, seq_stds = compute_etho_scores(Path(args.gt_dir), Path(args.inf_dir))
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
    scores = compute_cross_scores(preds, gt, cfg_path)
    _print_scores("Neuro scores", scores["neuro_scores"])
    _print_scores("Behavior scores", scores["behavior_scores"])
    _print_scores("Cross-modal scores", scores["cross_scores"])
    print(f"\nComposite neuro:  {scores['neuro_composite']:.3f}")
    print(f"Composite etho:   {scores['etho_composite']:.3f}")
    print(f"Composite cross:  {scores['cross_composite']:.3f}")
    print(f"Final composite:  {scores['composite']:.3f}")
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(scores, indent=2))
        print(f"Saved scores to {out}")


def _run_cross_full(args: argparse.Namespace) -> None:
    gt = _prompt_for_file("ground-truth", "gt_", args.gt)
    preds = _prompt_for_file("inference", "inference_", args.preds)
    cfg_path = _prompt_for_config(args.config)
    outdir = run_cross_full_analysis(preds, gt, cfg_path, output_root=args.output_root)
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
