
import math
import numpy as np
import pytest
from scipy import sparse

from vectors import (
    get_vector,
    get_sparse_vector,
    add,
    scalar_multiplication,
    linear_combination,
    dot_product,
    norm,
    distance,
    cos_between_vectors,
    is_orthogonal,
    solves_linear_systems,
)



def test_get_vector_shape_and_dtype():
    v = get_vector(5)
    assert isinstance(v, np.ndarray)
    assert v.shape == (5, 1)
    assert v.dtype.kind in {"f", "c"}

def test_get_vector_invalid_dim():
    with pytest.raises(ValueError):
        get_vector(0)
    with pytest.raises(ValueError):
        get_vector(-3)



def test_get_sparse_vector_basic():
    sv = get_sparse_vector(10)
    assert isinstance(sv, sparse.coo_matrix)
    assert sv.shape == (10, 1)
    assert 1 <= sv.nnz <= 10
    assert np.all(sv.col == 0)
    assert np.all((sv.row >= 0) & (sv.row < 10))

def test_get_sparse_vector_invalid_dim():
    with pytest.raises(ValueError):
        get_sparse_vector(0)



def test_add_column_vectors():
    x = np.array([[1.0], [2.0], [3.0]])
    y = np.array([[4.0], [5.0], [6.0]])
    res = add(x, y)
    np.testing.assert_allclose(res, np.array([[5.0], [7.0], [9.0]]))

def test_add_with_1d_inputs():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    res = add(x, y)
    np.testing.assert_allclose(res, np.array([[5.0], [7.0], [9.0]]))

def test_add_shape_mismatch():
    x = np.array([[1.0], [2.0]])
    y = np.array([[3.0], [4.0], [5.0]])
    with pytest.raises(ValueError):
        add(x, y)



def test_scalar_multiplication_basic():
    x = np.array([1.0, -2.0, 3.5])
    a = -2.0
    res = scalar_multiplication(x, a)
    np.testing.assert_allclose(res, np.array([[-2.0], [4.0], [-7.0]]))



def test_linear_combination_basic():
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([0.0, -1.0, 1.0])
    v3 = np.array([2.0, 0.0, 1.0])
    coeffs = [2.0, -1.0, 0.5]
    res = linear_combination([v1, v2, v3], coeffs)
    # Expected: 2*v1 - 1*v2 + 0.5*v3
    expected = 2*v1 - v2 + 0.5*v3
    np.testing.assert_allclose(res.reshape(-1), expected)

def test_linear_combination_errors():
    v1 = np.array([1.0, 2.0])
    v2 = np.array([3.0, 4.0, 5.0])
    with pytest.raises(ValueError):
        linear_combination([], [])
    with pytest.raises(ValueError):
        linear_combination([v1], [1.0, 2.0])
    with pytest.raises(ValueError):
        linear_combination([v1, v2], [1.0, 2.0])  # dimension mismatch



def test_dot_product_basic():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([-1.0, 0.5, 2.0])
    val = dot_product(x, y)
    assert isinstance(val, float)
    np.testing.assert_allclose(val, 1.0* -1.0 + 2.0*0.5 + 3.0*2.0)

def test_dot_product_shape_mismatch():
    with pytest.raises(ValueError):
        dot_product(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))



def test_norm_l1_l2_linf():
    x = np.array([3.0, -4.0, 0.0])
    assert math.isclose(norm(x, 1), 7.0)
    assert math.isclose(norm(x, 2), 5.0)
    assert math.isclose(norm(x, np.inf), 4.0)

def test_norm_invalid_order():
    with pytest.raises(ValueError):
        norm(np.array([1.0, 2.0]), 3)



def test_distance_l2():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 6.0, 3.0])
    d = distance(x, y)
    # sqrt((3)^2 + (4)^2 + 0^2) = 5
    assert math.isclose(d, 5.0)

def test_distance_shape_mismatch():
    with pytest.raises(ValueError):
        distance(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))



def test_cos_between_vectors_known_angles():
    x = np.array([1.0, 0.0])
    y = np.array([2.0, 0.0])
    a0 = cos_between_vectors(x, y)
    assert math.isclose(a0, 0.0, abs_tol=1e-12)

    x = np.array([1.0, 0.0])
    y = np.array([-3.0, 0.0])
    a180 = cos_between_vectors(x, y)
    assert math.isclose(a180, 180.0, abs_tol=1e-12)

    x = np.array([1.0, 0.0])
    y = np.array([0.0, 5.0])
    a90 = cos_between_vectors(x, y)
    assert math.isclose(a90, 90.0, abs_tol=1e-12)

def test_cos_between_vectors_zero_vector_error():
    with pytest.raises(ValueError):
        cos_between_vectors(np.array([0.0, 0.0]), np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        cos_between_vectors(np.array([1.0, 0.0]), np.array([0.0, 0.0]))

def test_cos_between_vectors_shape_mismatch():
    with pytest.raises(ValueError):
        cos_between_vectors(np.array([1.0, 0.0]), np.array([1.0, 0.0, 0.0]))



def test_is_orthogonal_true_cases():
    assert is_orthogonal(np.array([0.0, 0.0]), np.array([1.0, 2.0]))
    assert is_orthogonal(np.array([1.0, 2.0]), np.array([0.0, 0.0]))
    assert is_orthogonal(np.array([1.0, 0.0]), np.array([0.0, 3.0]))
    assert is_orthogonal(np.array([1.0, 0.0]), np.array([1e-12, 1.0]))

def test_is_orthogonal_false_case():
    assert not is_orthogonal(np.array([1.0, 2.0]), np.array([2.0, 1.0]))

def test_is_orthogonal_shape_mismatch():
    with pytest.raises(ValueError):
        is_orthogonal(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))



def test_solves_linear_systems_square_invertible():
    A = np.array([[3.0, 1.0], [2.0, 4.0]])
    b = np.array([7.0, 10.0])
    x = solves_linear_systems(A, b)
    assert x.shape == (2, 1)
    np.testing.assert_allclose(A @ x.reshape(-1), b)

def test_solves_linear_systems_square_singular_fallback_to_lstsq():
    A = np.array([[1.0, 2.0], [2.0, 4.0]])
    b = np.array([3.0, 6.0])
    x = solves_linear_systems(A, b)

    x_lstsq, *_ = np.linalg.lstsq(A, b, rcond=None)
    np.testing.assert_allclose(x.reshape(-1), x_lstsq, atol=1e-10)

def test_solves_linear_systems_overdetermined():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 3))
    true_x = np.array([1.5, -2.0, 0.5])
    b = A @ true_x + rng.normal(scale=1e-3, size=6)
    x = solves_linear_systems(A, b)
    x_lstsq, *_ = np.linalg.lstsq(A, b, rcond=None)
    np.testing.assert_allclose(x.reshape(-1), x_lstsq, atol=1e-8)

def test_solves_linear_systems_underdetermined():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(3, 5))
    b = rng.normal(size=3)
    x = solves_linear_systems(A, b)
    x_lstsq, *_ = np.linalg.lstsq(A, b, rcond=None)
    np.testing.assert_allclose(x.reshape(-1), x_lstsq, atol=1e-8)

def test_solves_linear_systems_shape_errors():
    with pytest.raises(ValueError):
        solves_linear_systems(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        solves_linear_systems(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0]))
