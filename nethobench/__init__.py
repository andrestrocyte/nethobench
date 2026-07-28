from nethobench.cross.pipeline import compute_cross_scores, run_cross_full_analysis
from nethobench.etho.pipeline import compute_etho_scores, run_etho_full_analysis
from nethobench.neuro.fidelity import compute_fidelity_scores
from nethobench.neuro.pipeline import compute_neuro_scores, run_neuro_full_analysis
from nethobench.synthetic.validation import (
    DEFAULT_PERTURBATIONS,
    PerturbationSpec,
    SyntheticDataset,
    SyntheticNeuralSpec,
    dataset_to_sequence_frame,
    generate_synthetic_neural_dataset,
    run_synthetic_neuro_validation,
)
from nethobench.synthetic.biophysical import (
    DEFAULT_BIOPHYSICAL_PERTURBATIONS,
    BiophysicalSyntheticNeuralSpec,
    generate_biophysical_synthetic_bundle,
    generate_biophysical_synthetic_dataset,
    run_biophysical_synthetic_neuro_validation,
)

__all__ = [
    "compute_neuro_scores",
    "compute_fidelity_scores",
    "run_neuro_full_analysis",
    "compute_etho_scores",
    "run_etho_full_analysis",
    "run_ethobench_notebook",
    "compute_cross_scores",
    "run_cross_full_analysis",
    "DEFAULT_PERTURBATIONS",
    "DEFAULT_BIOPHYSICAL_PERTURBATIONS",
    "PerturbationSpec",
    "SyntheticDataset",
    "SyntheticNeuralSpec",
    "BiophysicalSyntheticNeuralSpec",
    "dataset_to_sequence_frame",
    "generate_biophysical_synthetic_bundle",
    "generate_synthetic_neural_dataset",
    "generate_biophysical_synthetic_dataset",
    "run_synthetic_neuro_validation",
    "run_biophysical_synthetic_neuro_validation",
]


def run_ethobench_notebook(*args, **kwargs):
    """Backward-compatible alias for the native behavioral analysis pipeline."""
    return run_etho_full_analysis(*args, **kwargs)
