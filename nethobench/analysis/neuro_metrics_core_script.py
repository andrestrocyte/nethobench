from __future__ import annotations

from pathlib import Path

import nbformat


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "neuro_metrics.ipynb"

# Execute only the notebook cells needed for baseline metrics + final composite.
# Metric cells that also contain corruption sweeps are sliced before the sweep block.
CELL_SPECS = [
    {"marker": "import pandas as pd"},
    {
        "marker": "# === Distribution realism (KL only): baseline + corruption degradation ===",
        "stop_before": "# Same corruption families as previous sweep",
    },
    {
        "marker": "# === Mean difference metric (Top-10% regions) ===",
        "stop_before": "rows = [{",
        "append": """
mean_top10_df = pd.DataFrame([{
    'corruption': 'baseline',
    'magnitude': 0.0,
    'D_top10_mean': baseline['D'],
    'Mean_score01': baseline['Mean_score01'],
}])
""",
    },
    {
        "marker": "# --- Mutual information realism (STRICT, simple, benchmark-friendly) ---",
        "stop_before": "# B) Additive noise sweep",
    },
    {
        "marker": "# === Error / fidelity metric (simple, strict, benchmark-friendly) ===",
        "stop_before": "sigmas = [0.0",
    },
    {
        "marker": "# === Quantile / tail realism (simple, strict, benchmark-friendly) ===",
        "stop_before": "# Tail corruption sweep (merged)",
    },
    {
        "marker": "# === FC realism (simple, strict, benchmark-friendly) ===",
        "stop_before": "# 1) Region mixing noise",
    },
    {
        "marker": "# === PCA realism (simple, strict, benchmark-friendly) ===",
        "stop_before": "# 1) Region permutation",
    },
    {
        "marker": "# === AUTOCORR realism (improved, more sensitive, still simple/strict) ===",
        "stop_before": "rows = [{",
    },
    {
        "marker": "# === CROSSCORR realism (more sensitive, still simple/strict/benchmark-friendly) ===",
        "stop_before": "rows = [{",
    },
    {
        "marker": "# === Higher-order moments realism (simple, strict, benchmark-friendly) ===",
        "stop_before": "# 1) Tail spikes",
    },
    {
        "marker": "# === Corr-connectivity GRAPH realism (simple, strict, benchmark-friendly) ===",
        "stop_before": "perm_levels = [",
    },
    {
        "marker": "# === CCA realism (stricter, nested-q10, benchmark-friendly) ===",
        "stop_before": "# 1) Temporal shuffle",
    },
    {
        "marker": "# === MANIFOLD realism (simple geometry, strict, benchmark-friendly) ===",
    },
    {
        "marker": "# === TRAJECTORY DISTRIBUTION realism (FIXED: global GT-PCA basis, pooled across sequences) ===",
        "stop_before": "# -------------------------------------------------\n# 6) Corruption sweep",
    },
    {
        "marker": "# === BANDPOWER realism (simple, strict, benchmark-friendly) ===",
    },
    {
        "marker": "# === FINAL COMPOSITE (family-based, using exact upstream notebook result names) ===",
    },
]


preds_fname = globals().get("preds_fname")
gt_fname = globals().get("gt_fname")
ddconfig_path = globals().get("ddconfig_path")
SAVE_PLOTS_DIR = globals().get("SAVE_PLOTS_DIR")
WRAPPED_NOTEBOOK_PATH = globals().get("WRAPPED_NOTEBOOK_PATH")
ENABLE_PLOTS = bool(globals().get("ENABLE_PLOTS", False))

if preds_fname is None or gt_fname is None:
    raise ValueError("preds_fname and gt_fname must be provided")

if not NOTEBOOK_PATH.is_file():
    raise FileNotFoundError(f"Neuro notebook missing at {NOTEBOOK_PATH}")

try:
    import matplotlib

    if not ENABLE_PLOTS:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if SAVE_PLOTS_DIR:
        plot_dir = Path(SAVE_PLOTS_DIR)
        plot_dir.mkdir(parents=True, exist_ok=True)
        _plot_counter = {"n": 0}

        def _save_then_close_show(*args, **kwargs):
            figs = [plt.figure(num) for num in plt.get_fignums()]
            for fig in figs:
                _plot_counter["n"] += 1
                fig.savefig(
                    plot_dir / f"figure_{_plot_counter['n']:03d}.png",
                    dpi=200,
                    bbox_inches="tight",
                )
            plt.close("all")

        plt.show = _save_then_close_show
    elif not ENABLE_PLOTS:
        plt.show = lambda *args, **kwargs: plt.close("all")
except Exception:
    pass


try:
    display
except NameError:
    def display(*args, **kwargs):
        return None


def _patch_runtime_lines(source: str) -> str:
    lines = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("%") or stripped.startswith("!"):
            continue
        if stripped.startswith("preds_fname ="):
            line = f'preds_fname = r"{preds_fname}"'
        elif stripped.startswith("gt_fname ="):
            line = f'gt_fname = r"{gt_fname}"'
        elif stripped.startswith("ddconfig_path ="):
            line = f'ddconfig_path = r"{ddconfig_path}"'
        lines.append(line)
    return "\n".join(lines)


def _slice_cell_source(source: str, stop_before: str | None) -> str:
    patched = _patch_runtime_lines(source)
    if stop_before and stop_before in patched:
        patched = patched.split(stop_before, 1)[0].rstrip()
    return patched


def _resolve_code_cell(nb, marker: str):
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if source.lstrip().startswith(marker):
            return cell
    raise ValueError(f"Could not find notebook code cell starting with marker: {marker}")


nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
selected_cells = []

for spec in CELL_SPECS:
    cell = _resolve_code_cell(nb, spec["marker"])
    src = _slice_cell_source(cell.source, spec.get("stop_before"))
    append = spec.get("append")
    if append:
        src = f"{src.rstrip()}\n\n{append.strip()}\n"
    if src.strip():
        selected_cells.append(nbformat.v4.new_code_cell(src))
        exec(src, globals())


if WRAPPED_NOTEBOOK_PATH:
    wrapped_path = Path(WRAPPED_NOTEBOOK_PATH)
    wrapped_path.parent.mkdir(parents=True, exist_ok=True)
    wrapped_nb = nbformat.v4.new_notebook(cells=selected_cells, metadata=nb.metadata)
    with wrapped_path.open("w", encoding="utf-8") as fh:
        nbformat.write(wrapped_nb, fh)
