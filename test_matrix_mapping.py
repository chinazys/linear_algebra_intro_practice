
import numpy as np
import pytest

from matrix_mapping import (
    negative_matrix,
    reverse_matrix,
    affine_transform,
)



def test_negative_matrix_vector_and_matrix():
    v = np.array([1.0, -2.0, 3.0])
    m = np.array([[1.0, -2.0], [3.5, 0.0]])
    np.testing.assert_allclose(negative_matrix(v), -v)
    np.testing.assert_allclose(negative_matrix(m), -m)

def test_negative_matrix_invalid_ndim():
    with pytest.raises(ValueError):
        negative_matrix(np.zeros((2,2,2)))



def test_reverse_matrix_vector():
    v = np.array([1, 2, 3, 4])
    np.testing.assert_array_equal(reverse_matrix(v), np.array([4, 3, 2, 1]))

def test_reverse_matrix_matrix():
    X = np.array([[1, 2, 3],
                  [4, 5, 6]])
    expected = X[::-1, ::-1]
    np.testing.assert_array_equal(reverse_matrix(X), expected)

def test_reverse_matrix_invalid_ndim():
    with pytest.raises(ValueError):
        reverse_matrix(np.zeros((2,2,2)))



def _build_affine(alpha_deg, scale, shear, translate):
    alpha = np.deg2rad(alpha_deg)
    c, s = np.cos(alpha), np.sin(alpha)
    sx, sy = scale
    kx, ky = shear
    tx, ty = translate
    R = np.array([[ c, -s, 0.0],
                  [ s,  c, 0.0],
                  [0.0, 0.0, 1.0]])
    S = np.array([[sx, 0.0, 0.0],
                  [0.0, sy, 0.0],
                  [0.0, 0.0, 1.0]])
    Sh = np.array([[1.0, kx, 0.0],
                   [ky, 1.0, 0.0],
                   [0.0, 0.0, 1.0]])
    T = np.array([[1.0, 0.0, tx],
                  [0.0, 1.0, ty],
                  [0.0, 0.0, 1.0]])
    return T @ Sh @ S @ R

def test_affine_transform_identity_noop():
    P = np.array([[0.0, 1.0, -1.0],
                  [0.0, 2.0,  3.0]])
    out = affine_transform(P, alpha_deg=0.0, scale=(1.0,1.0), shear=(0.0,0.0), translate=(0.0,0.0))
    np.testing.assert_allclose(out, P, atol=1e-12)

def test_affine_transform_rotation_preserves_radius():
    theta = np.linspace(0, 2*np.pi, 8, endpoint=False)
    P = np.vstack([np.cos(theta), np.sin(theta)])  # (2,m)
    out = affine_transform(P, alpha_deg=37.0, scale=(1.0,1.0), shear=(0.0,0.0), translate=(0.0,0.0))
    r_in = np.sqrt(np.sum(P**2, axis=0))
    r_out = np.sqrt(np.sum(out**2, axis=0))
    np.testing.assert_allclose(r_out, r_in, atol=1e-12)

def test_affine_transform_scaling_and_translation_row_oriented_input():
    P = np.array([[1.0, 2.0],
                  [-1.0, 0.5],
                  [0.0, 0.0]])
    out = affine_transform(P, alpha_deg=0.0, scale=(2.0, 3.0), shear=(0.0, 0.0), translate=(10.0, -2.0))
    expected = np.column_stack([2.0*P[:,0] + 10.0, 3.0*P[:,1] - 2.0])
    np.testing.assert_allclose(out, expected, atol=1e-12)
    assert out.shape == P.shape

def test_affine_transform_combined_matches_manual_matrix():
    P = np.array([[2.0, -1.0, 0.5],
                  [1.0,  0.0, 3.0]])
    A = _build_affine(alpha_deg=30.0, scale=(1.5, 0.5), shear=(0.2, -0.1), translate=(3.0, -2.0))
    Ph = np.vstack([P, np.ones((1, P.shape[1]))])
    expected = (A @ Ph)[:2, :]
    out = affine_transform(P, alpha_deg=30.0, scale=(1.5, 0.5), shear=(0.2, -0.1), translate=(3.0, -2.0))
    np.testing.assert_allclose(out, expected, atol=1e-12)

def test_affine_transform_invalid_shapes():
    with pytest.raises(ValueError):
        affine_transform(np.array([1.0, 2.0, 3.0]), 0.0, (1.0,1.0), (0.0,0.0), (0.0,0.0))
    with pytest.raises(ValueError):
        affine_transform(np.zeros((3,3)), 0.0, (1.0,1.0), (0.0,0.0), (0.0,0.0))
    with pytest.raises(ValueError):
        affine_transform(np.zeros((2,2,2)), 0.0, (1.0,1.0), (0.0,0.0), (0.0,0.0))