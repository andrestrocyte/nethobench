from __future__ import annotations

from pathlib import Path

import nbformat
import numpy as np
import pandas as pd


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "neuro_metrics.ipynb"

# Execute only the notebook cells needed for the instant metric subset.
CELL_SPECS = [
    {"marker": "import pandas as pd"},
    {
        "marker": "# === Distribution realism (KL only): baseline + corruption degradation ===",
        "stop_before": "# Same corruption families as previous sweep",
        "append": """
_instant_kl_score01 = float(dist_scores.get('KL_score01_avg', np.nan)) if isinstance(dist_scores, dict) else np.nan
""",
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
_instant_mean_score01 = float(baseline.get('Mean_score01', np.nan)) if isinstance(baseline, dict) else np.nan
""",
    },
    {
        "marker": "# --- Mutual information realism (STRICT, simple, benchmark-friendly) ---",
        "stop_before": "# B) Additive noise sweep",
        "append": """
_instant_mi_score01 = float(results_mi.get('scores', {}).get('MI_score01', np.nan)) if isinstance(results_mi, dict) else np.nan
""",
    },
    {
        "marker": "# === Error / fidelity metric (simple, strict, benchmark-friendly) ===",
        "stop_before": "sigmas = [0.0",
        "append": """
_instant_error_score01 = float(err_simple.get('scores', {}).get('Error_score01', np.nan)) if isinstance(err_simple, dict) else np.nan
""",
    },
    {
        "marker": "# === Quantile / tail realism (simple, strict, benchmark-friendly) ===",
        "stop_before": "# Tail corruption sweep (merged)",
        "append": """
_instant_qnt_score01 = float(qnt_simple.get('scores', {}).get('QNT_score01', np.nan)) if isinstance(qnt_simple, dict) else np.nan
""",
    },
    {
        "marker": "# === FC realism (simple, strict, benchmark-friendly) ===",
        "stop_before": "# 1) Region mixing noise",
        "append": """
_instant_fc_score01 = float(fc_simple.get('scores', {}).get('FC_score01', np.nan)) if isinstance(fc_simple, dict) else np.nan
""",
    },
    {
        "marker": "# === PCA realism (simple, strict, benchmark-friendly) ===",
        "stop_before": "# 1) Region permutation",
        "append": """
_instant_pca_score01 = float(pca_simple.get('scores', {}).get('PCA_score01', np.nan)) if isinstance(pca_simple, dict) else np.nan
""",
    },
    {
        "marker": "# === Higher-order moments realism (simple, strict, benchmark-friendly) ===",
        "stop_before": "# 1) Tail spikes",
        "append": """
_instant_mom_score01 = float(mom_simple.get('scores', {}).get('MOM_score01', np.nan)) if isinstance(mom_simple, dict) else np.nan
""",
    },
    {
        "marker": "# === Corr-connectivity GRAPH realism (simple, strict, benchmark-friendly) ===",
        "stop_before": "perm_levels = [",
        "append": """
_instant_graph_score01 = float(graph_simple.get('scores', {}).get('GRAPH_score01', np.nan)) if isinstance(graph_simple, dict) else np.nan
""",
    },
    {
        "marker": "# === BANDPOWER realism (simple, strict, benchmark-friendly) ===",
        "append": """
_instant_bp_score01 = float(bandpower_simple.get('scores', {}).get('BP_score01', np.nan)) if isinstance(bandpower_simple, dict) else np.nan
""",
    },
    {"marker": "# === FINAL COMPOSITE (family-based, using exact upstream notebook result names) ==="},
]


preds_fname = globals().get("preds_fname")
gt_fname = globals().get("gt_fname")
ddconfig_path = globals().get("ddconfig_path")

if preds_fname is None or gt_fname is None:
    raise ValueError("preds_fname and gt_fname must be provided")

if not NOTEBOOK_PATH.is_file():
    raise FileNotFoundError(f"Neuro notebook missing at {NOTEBOOK_PATH}")


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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


INSTANT_NOTEBOOK_SCORES = {
    "KL_score01": float(globals().get("_instant_kl_score01", np.nan)),
    "KL_or_JSD_score01": float(globals().get("_instant_kl_score01", np.nan)),
    "Mean_score01": float(globals().get("_instant_mean_score01", np.nan)),
    "MI_score01": float(globals().get("_instant_mi_score01", np.nan)),
    "Error_score01": float(globals().get("_instant_error_score01", np.nan)),
    "QNT_score01": float(globals().get("_instant_qnt_score01", np.nan)),
    "FC_score01": float(globals().get("_instant_fc_score01", np.nan)),
    "PCA_score01": float(globals().get("_instant_pca_score01", np.nan)),
    "MOM_score01": float(globals().get("_instant_mom_score01", np.nan)),
    "GRAPH_score01": float(globals().get("_instant_graph_score01", np.nan)),
    "BP_score01": float(globals().get("_instant_bp_score01", np.nan)),
}
