"""Scientific invariants for opt-in alternatives; default metrics stay pinned."""
import numpy as np
import pytest

from nethobench.diagnostics.implementation_sensitivity import (
    sequence_pairs, compute_relational_sensitivity, _directed_similarity,
    conditional_transition_agreement, compute_conditional_state_sensitivity,
)


def test_pairs_do_not_cross_boundaries_or_bridge_missing_samples():
    x = np.array([[[1.], [2.], [3.]], [[100.], [101.], [102.]]])
    start, end = sequence_pairs(x, 1)
    np.testing.assert_array_equal(start[:, 0], [1, 2, 100, 101])
    np.testing.assert_array_equal(end[:, 0], [2, 3, 101, 102])
    x[0, 1, 0] = np.nan
    start, end = sequence_pairs(x, 1)
    np.testing.assert_array_equal(start[:, 0], [100, 101])
    assert len(sequence_pairs(x, 4)[0]) == 0


@pytest.mark.parametrize("lag", [0, -1, 1.5, True])
def test_pair_lag_validation(lag):
    with pytest.raises(ValueError):
        sequence_pairs(np.ones((2, 10, 3)), lag)


def test_relational_sequence_and_channel_permutations():
    rng = np.random.default_rng(74021)
    x = rng.normal(size=(5, 30, 4))
    y = x + rng.normal(size=x.shape) * .2
    expected = compute_relational_sensitivity(x, y)
    for xx, yy in ((x[[4, 2, 0, 3, 1]], y[[4, 2, 0, 3, 1]]),
                   (x[:, :, [2, 0, 3, 1]], y[:, :, [2, 0, 3, 1]])):
        actual = compute_relational_sensitivity(xx, yy)
        for key in ("lagged_covariance_directed", "var1_directed"):
            np.testing.assert_allclose(expected[key], actual[key], atol=1e-12)
    assert expected["reference_pair_counts"] == [145, 140, 130]


def test_lower_triangle_directed_edge_is_not_discarded():
    g = np.array([[1., 3, 2], [4, 1, 5], [9, 7, 1]])
    p = g.copy()
    p[2, 0] = -20
    assert _directed_similarity(g, p) < _directed_similarity(g, g)


def test_conditional_agreement_is_separate_from_occupancy():
    a = np.array([0, 1, 0, 1])
    b = np.array([2, 2, 2, 2])
    # Different numbers of sequences must not be truncated by a paired zip.
    result = conditional_transition_agreement([a] * 7 + [b], [a] + [b] * 2, 3)
    assert result["score"] == pytest.approx(1.)
    assert result["reference_origin_states"] == 3
    assert result["missing_prediction_origin_states"] == 0


def test_unvisited_reference_origins_are_penalized():
    a, b = np.array([0, 0, 0]), np.array([1, 1, 1])
    result = conditional_transition_agreement([a, b], [a], 2)
    assert result["score"] == pytest.approx(.5)
    assert result["missing_prediction_origin_states"] == 1


def test_no_transition_support_is_undefined():
    result = conditional_transition_agreement([np.array([0])], [np.array([0])], 2)
    assert np.isnan(result["score"])
    assert result["reference_origin_states"] == 0


def test_invalid_state_labels_are_rejected():
    with pytest.raises(ValueError):
        conditional_transition_agreement([np.array([0, 2])], [], 2)
    with pytest.raises(ValueError):
        conditional_transition_agreement([np.array([0., 1.])], [], 2)


def test_tensor_state_identity_and_finite_input_contract():
    x = np.random.default_rng(7).normal(size=(3, 20, 3))
    result = compute_conditional_state_sensitivity(x, x)
    assert all(v["score"] == pytest.approx(1.) for v in result["transitions"].values())
    assert not any(key.endswith("_score") for key in result)
    x[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compute_conditional_state_sensitivity(x, x)


def test_insufficient_tensor_state_support():
    result = compute_conditional_state_sensitivity(np.ones((1, 3, 2)), np.ones((1, 3, 2)))
    assert all(np.isnan(v["score"]) for v in result["transitions"].values())


def test_misaligned_tensors_rejected():
    with pytest.raises(ValueError, match="aligned"):
        compute_relational_sensitivity(np.ones((2, 3, 2)), np.ones((1, 3, 2)))


def test_all_missing_relational_input_is_undefined():
    x = np.full((2, 10, 3), np.nan)
    result = compute_relational_sensitivity(x, x)
    assert np.isnan(result["lagged_covariance_directed"])
    assert np.isnan(result["var1_directed"])
