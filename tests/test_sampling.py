import importlib

import numpy as np
import pandas as pd
import pytest

import clhs


implementation = importlib.import_module("clhs.clhs")


@pytest.fixture(autouse=True)
def restore_random_state():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def assert_partition(selected, remaining, original_selected, original_remaining):
    assert selected.shape == original_selected.shape
    assert remaining.shape == original_remaining.shape
    assert np.unique(selected).size == selected.size
    assert np.unique(remaining).size == remaining.size
    assert np.intersect1d(selected, remaining).size == 0
    np.testing.assert_array_equal(
        np.sort(np.concatenate([selected, remaining])),
        np.sort(np.concatenate([original_selected, original_remaining])),
    )


@pytest.mark.parametrize(
    "scores,selected,remaining,include,removable",
    [
        ([0, 9, 1], [30, 10, 20], [40, 50], None, [10]),
        ([0, 0, 9, 9], [30, 10, 20, 50], [40, 60], None, [20, 50]),
        ([0, 0, 9, 9], [30, 10, 20, 50], [40, 60], [30], [20, 50]),
        ([9, 8, 7, 6], [30, 10, 20, 50], [40, 60], [30, 20], [10]),
        ([0, 9, 1], [30, 10, 20], [40], [30, 20], [10]),
        ([0, 9, 1], [30, 10, 20], [40], 10, [20]),
        ([9], [30], [40], None, [30]),
        ([0, 0, 0], [30, 10, 20], [40], None, [30, 10, 20]),
    ],
)
def test_worst_swap(scores, selected, remaining, include, removable):
    selected = np.array(selected)
    remaining = np.array(remaining)
    scores = np.array(scores, dtype=float)
    original_selected = selected.copy()
    original_remaining = remaining.copy()
    original_scores = scores.copy()
    for seed in range(20):
        np.random.seed(seed)
        new_selected, new_remaining = clhs.resample_worst(
            continuous_objective_values=scores,
            sampled_indices=selected,
            remaining_indices=remaining,
            include=include,
        )
        removed = np.setdiff1d(selected, new_selected)
        assert removed.size == 1
        assert removed[0] in removable
        if include is not None:
            assert np.isin(include, new_selected).all()
        assert_partition(new_selected, new_remaining, selected, remaining)
        np.testing.assert_array_equal(selected, original_selected)
        np.testing.assert_array_equal(remaining, original_remaining)
        np.testing.assert_array_equal(scores, original_scores)


def test_worst_ties_can_choose_each_candidate():
    removed = set()
    for seed in range(20):
        np.random.seed(seed)
        selected, _ = clhs.resample_worst([0, 0, 9, 9], np.arange(4), np.array([4]))
        removed.update(set(range(4)) - set(selected))
    assert removed == {2, 3}


@pytest.mark.parametrize("method", ["random", "worst"])
@pytest.mark.parametrize(
    "selected,remaining,include,message",
    [
        ([10, 20], [], None, "remaining_indices must not be empty"),
        ([10, 20], [30], [10, 20], "No sampled indices are eligible"),
        ([], [30], None, "No sampled indices are eligible"),
    ],
)
def test_invalid_swap_inputs(method, selected, remaining, include, message):
    args = [np.array(selected, dtype=int), np.array(remaining, dtype=int)]
    if method == "worst":
        args.insert(0, np.zeros(len(selected)))
    with pytest.raises(ValueError, match=message):
        getattr(clhs, "resample_" + method)(*args, include=include)


@pytest.mark.parametrize("scores", [[], [1], [[1, 2]], [np.nan, 1], [1, np.inf]])
def test_invalid_worst_scores(scores):
    with pytest.raises(ValueError, match="continuous_objective_values"):
        clhs.resample_worst(scores, np.array([10, 20]), np.array([30]))


def test_random_swap_preserves_inputs_and_protected_indices():
    selected = np.array([30, 10, 20, 50])
    remaining = np.array([40, 60])
    new_selected, new_remaining = clhs.resample_random(selected, remaining, [30, 20])
    assert np.isin([30, 20], new_selected).all()
    assert_partition(new_selected, new_remaining, selected, remaining)
    np.testing.assert_array_equal(selected, [30, 10, 20, 50])
    np.testing.assert_array_equal(remaining, [40, 60])


def test_observation_scores_differ_from_stratum_objectives():
    values = np.array([3.5, 0.1, 0.2, 0.3])
    edges = np.arange(5, dtype=float)
    np.testing.assert_array_equal(clhs.continuous_objective_func(values, edges), [2, 1, 1, 0])
    scores = implementation._continuous_sample_scores(values, edges)
    np.testing.assert_array_equal(scores, [0, 2, 2, 2])
    selected, _ = clhs.resample_worst(scores, np.arange(4), np.array([4]))
    assert 0 in selected


@pytest.mark.parametrize(
    "values,edges,expected",
    [
        ([0, 0, 1, 2], [0, 0, 1, 2, 2], [1, 1, 0, 0]),
        ([0, 1, 2, 3], [0, 1, 2, 3, 3], [0, 0, 0, 0]),
        ([1, 1, 1], [1, 1, 1, 1], [2, 2, 2]),
    ],
)
def test_observation_scores_at_repeated_bin_edges(values, edges, expected):
    np.testing.assert_array_equal(
        implementation._continuous_sample_scores(values, edges), expected
    )


def test_observation_scores_sum_across_predictors():
    values = np.array([[0.2, 0.2], [0.3, 0.3], [0.4, 1.2], [3.2, 3.2]])
    edges = np.tile(np.arange(5, dtype=float)[:, None], (1, 2))
    np.testing.assert_array_equal(
        implementation._continuous_sample_scores(values, edges), [3, 3, 2, 0]
    )


@pytest.mark.parametrize("as_array", [False, True])
def test_categorical_objective_counts_missing_categories(as_array):
    values = pd.DataFrame({"kind": ["a", "a"]})
    if as_array:
        values = values.to_numpy()
    assert clhs.categorical_objective_func(values, {"kind": pd.Series({"b": 0.5, "a": 0.5})}) == 1


def test_categorical_objective_aligns_each_predictor():
    values = pd.DataFrame({"kind": ["a", "a"], "code": [2, 3]})
    proportions = {"kind": pd.Series({"a": 0.5, "b": 0.5}), "code": pd.Series({3: 0.5, 2: 0.5})}
    assert clhs.categorical_objective_func(values, proportions) == 1


def predictors_for(kind):
    rng = np.random.default_rng(0)
    if kind in ("float32", "float64"):
        return pd.DataFrame(rng.random((20, 3)).astype(kind), columns=["x1", "x2", "x3"])
    if kind == "integer":
        return pd.DataFrame({"code": np.arange(20) % 3, "other": np.arange(20) % 2})
    if kind == "mixed":
        return pd.DataFrame({"x": rng.random(20), "code": np.arange(20) % 3})
    return pd.DataFrame({"x": np.repeat(np.arange(5, dtype=float), 4), "y": rng.random(20)})


@pytest.mark.parametrize("kind", ["float32", "float64", "integer", "mixed", "tied"])
@pytest.mark.parametrize("progress", [False, True])
def test_public_sampler(kind, progress):
    predictors = predictors_for(kind)
    original = predictors.copy(deep=True)
    mask = np.ones(20, dtype=bool)
    mask[[2, 5, 13]] = False
    kwargs = dict(num_samples=10, max_iterations=100, random_state=42, progress=progress, good_mask=mask, include=[0, 7])
    result = clhs.clhs(predictors, **kwargs)
    repeated = clhs.clhs(predictors, **kwargs)
    np.testing.assert_array_equal(result["sample_indices"], repeated["sample_indices"])
    assert np.isfinite(result["obj"])
    assert np.isfinite(result["obj_continuous"]).all()
    assert np.isin([0, 7], result["sample_indices"]).all()
    valid = np.flatnonzero(mask)
    assert_partition(result["sample_indices"], result["remaining_indices"], valid[:10], valid[10:])
    pd.testing.assert_frame_equal(result["x"], predictors.iloc[result["sample_indices"]])
    pd.testing.assert_frame_equal(result["r"], predictors.iloc[result["remaining_indices"]])
    pd.testing.assert_frame_equal(predictors, original)
    continuous = predictors.select_dtypes(include=["float32", "float64", float])
    categorical = predictors.select_dtypes(exclude=["float32", "float64", float])
    kappa = {col: categorical.loc[mask, col].value_counts(normalize=True) for col in categorical}
    recomputed = clhs.clhs_objective_func(
        continuous.iloc[result["sample_indices"]], categorical.iloc[result["sample_indices"]],
        clhs.get_strata(continuous, 10, mask), kappa,
        clhs.get_correlation_matrix(continuous, mask),
    )
    assert result["obj"] == pytest.approx(recomputed[0])
    np.testing.assert_array_equal(result["obj_continuous"], recomputed[1])
    assert result["obj_continuous"].size == (0 if kind == "integer" else 10)


def test_report_public_sampler_fixture():
    result = clhs.clhs(predictors_for("float64"), num_samples=10, max_iterations=100, random_state=42, progress=False)
    assert np.isfinite(result["obj"])
    assert_partition(result["sample_indices"], result["remaining_indices"], np.arange(10), np.arange(10, 20))


def test_sampler_passes_observation_scores_from_accepted_sample(monkeypatch):
    real_resample = implementation.resample
    predictors = predictors_for("float64")
    quantiles = clhs.get_strata(predictors, 10)
    calls = []

    def checked_resample(p, sampled, remaining, scores, include=None):
        np.testing.assert_array_equal(scores, implementation._continuous_sample_scores(predictors.iloc[sampled], quantiles))
        calls.append(sampled.copy())
        return real_resample(p, sampled, remaining, scores, include)

    monkeypatch.setattr(implementation, "resample", checked_resample)
    clhs.clhs(predictors, 10, max_iterations=100, random_state=42, progress=False)
    assert len(calls) == 100
    assert any(np.array_equal(a, b) for a, b in zip(calls, calls[1:]))
