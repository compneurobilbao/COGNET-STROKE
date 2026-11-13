"""Unit tests of compute_cdr and compute_overlap, including the edge cases."""

import warnings

import numpy as np
import pytest

from cognet_runtime import cdr_flag, compute_cdr, compute_overlap


def original_cdr(concept_mask, disconection_map, nulls=None):
    """Verbatim logic of the analysis workflow's implementation (before the runtime split)."""
    inside = np.mean(disconection_map[concept_mask == 1])
    outside = np.mean(disconection_map[concept_mask == -1])
    value = inside if outside == 0 else inside / outside
    if nulls is None:
        return value
    vals = []
    for null in nulls:
        i = np.mean(disconection_map[null == 1])
        o = np.mean(disconection_map[null == -1])
        vals.append(i if o == 0 else i / o)
    vals = np.array(vals)
    hits = len(vals[vals > value])
    return value, (1 / len(vals) if hits == 0 else hits / len(vals))


def test_hand_computed_ratio():
    mask = np.array([1, 1, -1, -1, -1])
    dis = np.array([4.0, 2.0, 1.0, 1.0, 1.0])
    assert compute_cdr(mask, dis) == pytest.approx(3.0)  # mean 3 inside, 1 outside


def test_zero_outside_mean_returns_inside_mean():
    mask = np.array([1, 1, -1, -1])
    dis = np.array([2.0, 4.0, 0.0, 0.0])
    assert compute_cdr(mask, dis) == pytest.approx(3.0)
    assert cdr_flag(mask, dis) == "zero_outside_mean"
    assert cdr_flag(mask, np.array([2.0, 4.0, 1.0, 1.0])) is None


def test_zero_outside_and_zero_inside():
    mask = np.array([1, -1])
    assert compute_cdr(mask, np.zeros(2)) == 0.0


@pytest.mark.parametrize(
    "mask",
    [np.array([-1, -1, -1]), np.array([1, 1, 1]), np.array([0, 0, 0]), np.array([1, 0, 0])],
)
def test_empty_inside_or_outside_is_nan_without_warnings(mask):
    dis = np.array([1.0, 2.0, 3.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(compute_cdr(mask, dis))
        value, p = compute_cdr(mask, dis, np.array([[1, -1, 1]]))
    assert np.isnan(value) and np.isnan(p)
    assert cdr_flag(mask, dis) == "empty_mask"


def test_parcels_with_value_zero_are_ignored():
    dis = np.array([5.0, 3.0, 100.0, 1.0])
    assert compute_cdr(np.array([1, 0, 0, -1]), dis) == pytest.approx(5.0)


def test_p_value_counts_strictly_larger_nulls():
    mask = np.array([1, 1, -1, -1, -1, -1])
    dis = np.array([3.0, 3.0, 1.0, 1.0, 1.0, 1.0])  # observed CDR 3
    nulls = np.array(
        [
            [1, -1, 1, -1, -1, -1],  # inside {3, 1}: 2/1 = 2
            [1, 1, -1, -1, -1, -1],  # identical to the observed mask: 3, a tie, not a hit
            [-1, -1, 1, 1, -1, -1],  # 1
            [-1, -1, -1, -1, 1, 1],  # 1
        ]
    )
    value, p = compute_cdr(mask, dis, nulls)
    assert value == pytest.approx(3.0)
    assert p == pytest.approx(0.25)  # no hit: floor 1 / n_nulls


def test_p_value_with_hits():
    mask = np.array([1, -1, -1, -1])
    dis = np.array([1.0, 1.0, 1.0, 1.0])  # observed 1
    nulls = np.array([[-1, 1, -1, -1]] * 2 + [[1, -1, -1, -1]] * 2)  # all equal 1: no hits
    assert compute_cdr(mask, dis, nulls)[1] == pytest.approx(0.25)
    dis = np.array([1.0, 1.0, 2.0, 1.0])  # observed 1 / 1.33 = 0.75
    nulls = np.array([[-1, -1, 1, -1], [-1, -1, 1, -1], [1, -1, -1, -1]])
    value, p = compute_cdr(mask, dis, nulls)
    assert p == pytest.approx(2 / 3)  # two nulls (ratio 2) exceed 0.75; the third ties


def test_p_value_never_below_one_over_n():
    mask = np.array([1, -1, -1])
    dis = np.array([10.0, 1.0, 1.0])
    nulls = np.array([[-1, 1, -1]] * 50)
    assert compute_cdr(mask, dis, nulls)[1] == pytest.approx(1 / 50)


def test_nan_null_is_not_a_hit_but_counts_in_the_denominator():
    mask = np.array([1, -1, -1])
    dis = np.array([1.0, 2.0, 2.0])  # observed 0.5
    nulls = np.array([[1, 1, 1], [-1, 1, -1], [1, -1, -1]])  # null 1: no outside -> NaN
    value, p = compute_cdr(mask, dis, nulls)
    # null 2: inside 2 / outside 1.5 = 1.33 (hit); null 3: 1 / 2 = 0.5 (tie)
    assert p == pytest.approx(1 / 3)


def test_matches_the_original_implementation_on_random_data():
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = 60
        mask = np.where(rng.random(n) > 0.7, 1, -1)
        dis = rng.random(n).astype(np.float32) * (rng.random(n) > 0.2)
        nulls = np.where(rng.random((200, n)) > 0.7, 1, -1)
        assert compute_cdr(mask, dis) == original_cdr(mask, dis)
        assert compute_cdr(mask, dis, nulls) == original_cdr(mask, dis, nulls)


def test_input_validation():
    with pytest.raises(ValueError):
        compute_cdr(np.array([1, -1]), np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        compute_cdr(np.ones((2, 2)), np.ones((2, 2)))
    with pytest.raises(ValueError):
        compute_cdr(np.array([1, -1]), np.array([1.0, 2.0]), np.ones((3, 5)))
    with pytest.raises(ValueError):
        compute_cdr(np.array([1, -1]), np.array([1.0, 2.0]), np.ones((0, 2)))


def test_overlap_hand_computed():
    mask = np.array([1, 1, -1, -1])
    lesion = np.array([1.0, 0.0, 1.0, 1.0])
    assert compute_overlap(mask, lesion) == pytest.approx(1 / 3)


def test_overlap_empty_lesion_is_nan():
    mask = np.array([1, -1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(compute_overlap(mask, np.zeros(2)))
        value, p = compute_overlap(mask, np.zeros(2), np.array([[1, -1]]))
    assert np.isnan(value) and np.isnan(p)


def test_overlap_p_value():
    mask = np.array([1, 1, -1, -1, -1])
    lesion = np.array([1.0, 1.0, 0.0, 0.0, 0.0])  # overlap 1: nothing exceeds it
    nulls = np.array([[1, -1, -1, -1, -1], [-1, -1, 1, 1, -1], [-1, -1, -1, -1, 1]])
    value, p = compute_overlap(mask, lesion, nulls)
    assert value == 1.0
    assert p == pytest.approx(1 / 3)
    value, p = compute_overlap(np.array([-1, -1, 1, 1, 1]), lesion, nulls)
    assert value == 0.0
    assert p == pytest.approx(1 / 3)  # only null 1 (overlap 0.5) exceeds 0
