"""Quantization and topographic error of aweSOM._batch_core, without JAX."""

import numpy as np
import pytest

from aweSOM._batch_core import (
    quantization_error,
    quantization_error_fast,
    topographic_error,
    topographic_error_fast,
)


def test_quality_measures_small_example():
    # 2x2 lattice on a line: nodes 0 (0,0), 1 (1,0), 2 (0,1), 3 (1,1)
    w = np.array([[0.0], [1.0], [10.0], [3.0]])
    x = np.array([[0.1], [0.9], [9.0], [2.0]])
    assert quantization_error(w, x) == pytest.approx((0.1 + 0.1 + 1.0 + 1.0) / 4)
    # BMU pairs: (0,1) adjacent, (1,0) adjacent, (2,3) diagonal, tie 1/3 at 2.0 -> (1,3) adjacent
    assert topographic_error(w, x, 2, 2) == 0.0
    w5 = np.array([[0.0], [5.0], [5.0], [0.5], [9.0], [9.0]])  # 3x2 lattice
    # BMU 0 (0,0), second 3 (0,1): adjacent; x=8 -> 4 (1,1) then 5 (2,1): adjacent
    # x=2.7: 3 (0,1) then 1 (1,0): adjacent (diagonal); x=4.0 -> 1 (1,0), 2 (2,0)
    assert topographic_error(w5, np.array([[0.1], [8.0], [2.7], [4.0]]), 3, 2) == 0.0
    w_far = np.array([[0.0], [9.0], [9.0], [9.0], [9.0], [0.5]])
    # BMU 0 (0,0), second 5 (2,1): Chebyshev distance 2 -> error
    assert topographic_error(w_far, np.array([[0.2]]), 3, 2) == 1.0
    # three-way tie between nodes 0, 1 and 5: (distance, index) order picks 0 and 1
    w_tie = np.array([[1.0], [-1.0], [9.0], [9.0], [9.0], [1.0]])
    assert topographic_error(w_tie, np.array([[0.0]]), 3, 2) == 0.0


@pytest.mark.parametrize("n_feat", [2, 4, 7, 10])
def test_fast_qe_te_equal_reference(n_feat):
    rng = np.random.default_rng(n_feat)
    w = rng.normal(size=(9 * 7, n_feat))
    x = rng.normal(size=(5000, n_feat))
    x[:50] = w[:50]  # exact hits
    x[50:60] = 0.5 * (w[0] + w[1])  # possible near-ties
    assert quantization_error_fast(w, x) == quantization_error(w, x)
    assert topographic_error_fast(w, x, 9, 7) == topographic_error(w, x, 9, 7)
