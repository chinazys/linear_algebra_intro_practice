
import math
import numpy as np
import pytest

from matrices import (
    get_matrix,
    add,
    scalar_multiplication,
    dot_product,
    identity_matrix,
    matrix_inverse,
    matrix_transpose,
    hadamard_product,
    basis,
    norm,
)



def test_get_matrix_shape_and_dtype():
    A = get_matrix(3, 4)
    assert isinstance(A, np.ndarray)
    assert A.shape == (3, 4)
    assert A.dtype.kind in {"f", "c"}

def test_get_matrix_invalid_dims():
    with pytest.raises(ValueError):
        get_matrix(0, 3)
    with pytest.raises(ValueError):
        get_matrix(3, 0)
    with pytest.raises(ValueError):
        get_matrix(-1, 2)



def test_add_basic():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    Y = np.array([[5.0, 6.0], [7.0, 8.0]])
    Z = add(X, Y)
    np.testing.assert_allclose(Z, np.array([[6.0, 8.0], [10.0, 12.0]]))

def test_add_shape_mismatch():
    X = np.zeros((2, 2))
    Y = np.zeros((3, 2))
    with pytest.raises(ValueError):
        add(X, Y)



def test_scalar_multiplication_basic():
    X = np.array([[1.0, -2.0], [0.5, 3.0]])
    out = scalar_multiplication(X, -2.0)
    np.testing.assert_allclose(out, np.array([[-2.0, 4.0], [-1.0, -6.0]]))



def test_dot_product_matrix_matrix():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    AB = dot_product(A, B)
    np.testing.assert_allclose(AB, A @ B)

def test_dot_product_matrix_vector():
    A = np.array([[2.0, 0.0, -1.0],
                  [1.0, 3.0,  4.0]])
    v = np.array([1.0, 2.0, 3.0])
    Av = dot_product(A, v)
    np.testing.assert_allclose(Av, A @ v)

def test_dot_product_bad_dims():
    with pytest.raises(ValueError):
        dot_product(np.array([1.0, 2.0]), np.array([[1.0], [2.0]]))  # x not 2D
    with pytest.raises(ValueError):
        dot_product(np.eye(2), np.zeros((2, 2, 2)))  # y not 1D/2D



def test_identity_matrix_basic():
    I = identity_matrix(4)
    np.testing.assert_allclose(I, np.eye(4))
    assert I.dtype == float

def test_identity_matrix_invalid():
    with pytest.raises(ValueError):
        identity_matrix(0)



def test_matrix_inverse_basic():
    A = np.array([[3.0, 1.0], [2.0, 5.0]])
    invA = matrix_inverse(A)
    np.testing.assert_allclose(invA @ A, np.eye(2), atol=1e-10)
    np.testing.assert_allclose(A @ invA, np.eye(2), atol=1e-10)

def test_matrix_inverse_singular():
    A = np.array([[1.0, 2.0], [2.0, 4.0]])  # singular
    with pytest.raises(ValueError):
        matrix_inverse(A)

def test_matrix_inverse_not_square():
    with pytest.raises(ValueError):
        matrix_inverse(np.ones((2,3)))



def test_matrix_transpose_basic():
    A = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
    AT = matrix_transpose(A)
    np.testing.assert_allclose(AT, A.T)

def test_matrix_transpose_invalid():
    with pytest.raises(ValueError):
        matrix_transpose(np.array([1.0, 2.0, 3.0]))  # not 2D



def test_hadamard_product_basic():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    Y = np.array([[5.0, 6.0], [7.0, 8.0]])
    H = hadamard_product(X, Y)
    np.testing.assert_allclose(H, X * Y)

def test_hadamard_product_shape_mismatch():
    with pytest.raises(ValueError):
        hadamard_product(np.zeros((2,2)), np.zeros((3,2)))



def test_basis_full_rank_square():
    A = np.array([[1.0, 0.0, 2.0],
                  [0.0, 1.0, 1.0],
                  [0.0, 0.0, 1.0]])
    idx = basis(A)
    assert len(idx) == 3
    assert set(idx) == {0,1,2}

def test_basis_rank_deficient():
    A = np.array([[1.0, 2.0, 3.0],
                  [0.0, 1.0, 1.0],
                  [1.0, 1.0, 2.0]])
    idx = basis(A)
    assert len(idx) == 2

    B = A[:, idx]
    assert np.linalg.matrix_rank(B) == 2

def test_basis_all_zero_matrix():
    A = np.zeros((4, 3))
    idx = basis(A)
    assert idx == tuple()

def test_basis_invalid_input():
    with pytest.raises(ValueError):
        basis(np.array([1.0, 2.0, 3.0]))



def test_norm_fro_spectral_inf():
    A = np.array([[3.0, 4.0],
                  [0.0, 0.0]])
    assert math.isclose(norm(A, "fro"), 5.0, rel_tol=1e-12, abs_tol=1e-12)

    s = np.linalg.svd(A, compute_uv=False)
    assert math.isclose(norm(A, 2), s.max(), rel_tol=1e-12, abs_tol=1e-12)

    row_sums = np.sum(np.abs(A), axis=1)
    assert math.isclose(norm(A, np.inf), row_sums.max(), rel_tol=1e-12, abs_tol=1e-12)

def test_norm_invalid_order():
    with pytest.raises(ValueError):
        norm(np.array([1.0, 2.0, 3.0]), "fro")
