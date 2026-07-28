from __future__ import annotations

from pathlib import Path

from nethobench.analysis.behavior_crossmodal_supplement import (
    run_behavior_crossmodal_supplement,
)
from nethobench.analysis.bulletproof_validation import (
    run_bulletproof_validation,
)
from nethobench.ibl_adapter import (
    build_ibl_repeated_site_dataset,
    discover_ibl_repeated_site_sessions,
    export_ibl_repeated_site_benchmark,
    run_ibl_expanded_study,
    score_ibl_study,
    train_ibl_study_models,
    write_ibl_manifest_template,
)
from nethobench.tribev2_adapter import (
    export_tribev2_benchmark,
    write_manifest_template,
)


def _run_tribev2_export(args) -> None:
    report = export_tribev2_benchmark(
        Path(args.manifest),
        output_root=args.output_root,
    )
    print(f"TRIBE v2 benchmark report: {report['outputs']['pred_parcels_csv']}")


def _run_tribev2_init_manifest(args) -> None:
    out = write_manifest_template(
        args.dataset,
        Path(args.output),
        name=args.name,
        tribev2_root=args.tribev2_root,
    )
    print(f"Wrote manifest template to {out}")


def _run_ibl_init_manifest(args) -> None:
    out = write_ibl_manifest_template(Path(args.output), name=args.name)
    print(f"Wrote IBL manifest template to {out}")


def _run_ibl_discover(args) -> None:
    report = discover_ibl_repeated_site_sessions(
        Path(args.manifest),
        output_root=args.output_root,
    )
    print(f"IBL discovery report: {report['report_json']}")


def _run_ibl_export(args) -> None:
    report = export_ibl_repeated_site_benchmark(
        Path(args.manifest),
        output_root=args.output_root,
        dry_run=args.dry_run,
    )
    print(f"IBL export report: {report.get('report_json', report)}")


def _run_ibl_build_dataset(args) -> None:
    report = build_ibl_repeated_site_dataset(
        Path(args.manifest),
        output_root=args.output_root,
        dry_run=args.dry_run,
    )
    print(f"IBL dataset report: {report.get('report_json', report)}")


def _run_ibl_train_models(args) -> None:
    report = train_ibl_study_models(
        Path(args.manifest),
        output_root=args.output_root,
        dry_run=args.dry_run,
    )
    print(f"IBL training report: {report.get('report_json', report)}")


def _run_ibl_score_study(args) -> None:
    report = score_ibl_study(
        Path(args.manifest),
        output_root=args.output_root,
        dry_run=args.dry_run,
    )
    print(f"IBL scoring report: {report.get('report_json', report)}")


def _run_ibl_run_study(args) -> None:
    report = run_ibl_expanded_study(
        Path(args.manifest),
        output_root=args.output_root,
        dry_run=args.dry_run,
    )
    print(f"IBL study report: {report.get('study_report_json', report)}")


def _run_bulletproof(args) -> None:
    result = run_bulletproof_validation(
        output_root=args.output_root,
        mode=args.mode,
        min_free_disk_gb=args.min_free_disk_gb,
        run_full_notebook_scores=args.run_full_notebook_scores,
    )
    print(f"Bulletproof validation report: {result['outputs']['bulletproof_analysis_report_json']}")


def _run_behavior_crossmodal(args) -> None:
    result = run_behavior_crossmodal_supplement(
        output_root=args.output_root,
        mode=args.mode,
        min_free_disk_gb=args.min_free_disk_gb,
    )
    print(
        "Behavior/cross-modal report: "
        f"{result['outputs']['behavior_crossmodal_supplement_report_json']}"
    )


def _add_output_root(parser) -> None:
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Optional output directory override.",
    )


def _add_ibl_manifest_and_output(parser, *, dry_run: bool = False) -> None:
    parser.add_argument("--manifest", required=True, help="IBL study manifest JSON.")
    _add_output_root(parser)
    if dry_run:
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Validate inputs without downloading data or training models.",
        )


def add_study_subparsers(subparsers) -> None:
    bulletproof = subparsers.add_parser(
        "bulletproof-run",
        help="Run the synthetic neural-validity stress test.",
    )
    bulletproof.add_argument("--mode", choices=["quick", "full"], default="quick")
    _add_output_root(bulletproof)
    bulletproof.add_argument("--min-free-disk-gb", type=float, default=8.0)
    bulletproof.add_argument(
        "--run-full-notebook-scores",
        action="store_true",
        help="Also run the slower notebook-backed scoring path.",
    )
    bulletproof.set_defaults(func=_run_bulletproof)

    behavior_crossmodal = subparsers.add_parser(
        "behavior-crossmodal-supp",
        help="Run the behavioral and neural-behavioral proof-of-concept study.",
    )
    behavior_crossmodal.add_argument("--mode", choices=["quick", "full"], default="quick")
    _add_output_root(behavior_crossmodal)
    behavior_crossmodal.add_argument("--min-free-disk-gb", type=float, default=4.0)
    behavior_crossmodal.set_defaults(func=_run_behavior_crossmodal)

    tribe_export = subparsers.add_parser(
        "tribev2-export",
        help="Run the TRIBE v2 cortical-realism export pipeline.",
    )
    tribe_export.add_argument("--manifest", required=True)
    _add_output_root(tribe_export)
    tribe_export.set_defaults(func=_run_tribev2_export)

    tribe_manifest = subparsers.add_parser(
        "tribev2-init-manifest",
        help="Write a TRIBE v2 benchmark manifest template.",
    )
    tribe_manifest.add_argument(
        "--dataset",
        required=True,
        choices=["lahner2024bold", "hcp", "narratives"],
    )
    tribe_manifest.add_argument("--output", required=True)
    tribe_manifest.add_argument("--name")
    tribe_manifest.add_argument("--tribev2-root")
    tribe_manifest.set_defaults(func=_run_tribev2_init_manifest)

    ibl_manifest = subparsers.add_parser(
        "ibl-init-manifest",
        help="Write an IBL repeated-site manifest template.",
    )
    ibl_manifest.add_argument("--output", required=True)
    ibl_manifest.add_argument("--name")
    ibl_manifest.set_defaults(func=_run_ibl_init_manifest)

    ibl_discover = subparsers.add_parser(
        "ibl-discover",
        help="Discover repeated-site IBL sessions.",
    )
    _add_ibl_manifest_and_output(ibl_discover)
    ibl_discover.set_defaults(func=_run_ibl_discover)

    commands = (
        ("ibl-export", "Export the compact IBL benchmark.", _run_ibl_export),
        ("ibl-build-dataset", "Build the expanded IBL dataset.", _run_ibl_build_dataset),
        ("ibl-train-models", "Train expanded IBL forecasting models.", _run_ibl_train_models),
        ("ibl-score-study", "Score expanded IBL predictions.", _run_ibl_score_study),
        ("ibl-run-study", "Run the expanded IBL study.", _run_ibl_run_study),
    )
    for name, help_text, handler in commands:
        command = subparsers.add_parser(name, help=help_text)
        _add_ibl_manifest_and_output(command, dry_run=True)
        command.set_defaults(func=handler)
