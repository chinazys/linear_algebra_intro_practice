
import numpy as np
import pytest

from matrix_decompostion import (
    lu_decomposition,
    qr_decomposition,
    determinant,
    eigen,
    svd,
)



def test_lu_square_reconstruction():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 4))
    P, L, U = lu_decomposition(A)

    assert P.shape == (4, 4)
    assert L.shape == (4, 4)
    assert U.shape == (4, 4)

    np.testing.assert_allclose(P @ P.T, np.eye(4), atol=1e-12)

    assert np.allclose(np.diag(L), 1.0)
    assert np.allclose(L, np.tril(L))

    assert np.allclose(U, np.triu(U))

    np.testing.assert_allclose(P @ A, L @ U, atol=1e-10)

def test_lu_rectangular_reconstruction():
    rng = np.random.default_rng(1)
    m, n = 5, 3
    A = rng.normal(size=(m, n))
    P, L, U = lu_decomposition(A)

    np.testing.assert_allclose(P @ A, L @ U, atol=1e-10)

    k = min(m, n)
    assert P.shape == (m, m)
    assert L.shape == (m, k)
    assert U.shape == (k, n)

    Lkk = L[:k, :k]
    assert np.allclose(np.diag(Lkk), 1.0)
    assert np.allclose(Lkk, np.tril(Lkk))

    Ukk = U[:k, :k]
    assert np.allclose(Ukk, np.triu(Ukk))



def test_qr_reconstruction_and_orthonormality():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(6, 4))
    Q, R = qr_decomposition(A)

    assert Q.shape == (6, 4)
    assert R.shape == (4, 4)

    np.testing.assert_allclose(Q.T @ Q, np.eye(4), atol=1e-10)
    np.testing.assert_allclose(Q @ R, A, atol=1e-10)



def test_determinant_square_and_property():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(3, 3))
    B = rng.normal(size=(3, 3))
    detA = determinant(A)
    detB = determinant(B)
    detAB = determinant(A @ B)
    np.testing.assert_allclose(detAB, detA * detB, atol=1e-10)

def test_determinant_non_square_raises():
    with pytest.raises(ValueError):
        determinant(np.ones((2, 3)))



def test_eigen_reconstruction_general():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(4, 4))
    w, V = eigen(A)

    assert w.shape == (4,)
    assert V.shape == (4, 4)

    np.testing.assert_allclose(A @ V, V @ np.diag(w), atol=1e-8)

def test_eigen_symmetric_real():
    rng = np.random.default_rng(5)
    M = rng.normal(size=(5, 5))
    A = 0.5 * (M + M.T)
    w, V = eigen(A)

    assert np.allclose(w.imag, 0.0, atol=1e-12)
    np.testing.assert_allclose(A @ V, V @ np.diag(w), atol=1e-8)

def test_eigen_non_square_raises():
    with pytest.raises(ValueError):
        eigen(np.ones((2, 3)))



def test_svd_reconstruction_and_ordering_square():
    rng = np.random.default_rng(6)
    A = rng.normal(size=(4, 4))
    U, s, V = svd(A)

    assert U.shape == (4, 4)
    assert s.shape == (4,)
    assert V.shape == (4, 4)

    S = np.diag(s)
    np.testing.assert_allclose(U @ S @ V.T, A, atol=1e-10)

    assert np.all(s >= -1e-12)
    assert np.all(s[:-1] >= s[1:] - 1e-12)

def test_svd_rectangular_tall_and_wide():
    rng = np.random.default_rng(7)
    Atall = rng.normal(size=(6, 3))
    U, s, V = svd(Atall)
    assert U.shape == (6, 3)
    assert s.shape == (3,)
    assert V.shape == (3, 3)
    np.testing.assert_allclose(U @ np.diag(s) @ V.T, Atall, atol=1e-10)

    Awide = rng.normal(size=(3, 6))
    U, s, V = svd(Awide)
    assert U.shape == (3, 3)
    assert s.shape == (3,)
    assert V.shape == (6, 3)
    np.testing.assert_allclose(U @ np.diag(s) @ V.T, Awide, atol=1e-10)
