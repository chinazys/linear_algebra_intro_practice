from typing import Sequence

import numpy as np
from scipy import sparse


# --- tests ---
# run `pytest test_vectors.py` to execute the tests


# --- utils ---
def _as_col(x: np.ndarray) -> np.ndarray:
    """Return x as a (n, 1) column vector (copy if needed)."""
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim == 2 and x.shape[1] == 1:
        return x
    if x.ndim == 2 and x.shape[0] == 1:
        return x.T
    return x.reshape(-1, 1)

def _as_1d(x: np.ndarray) -> np.ndarray:
    """Return x as a contiguous 1D array."""
    return np.asarray(x).reshape(-1)


# --- tasks ---
def get_vector(dim: int) -> np.ndarray:
    """Create random column vector with dimension dim.

    Args:
        dim (int): vector dimension.

    Returns:
        np.ndarray: column vector.
    """
    if dim <= 0:
        raise ValueError("dim must be positive.")
    rng = np.random.default_rng()
    return rng.normal(size=(dim, 1))


def get_sparse_vector(dim: int) -> sparse.coo_matrix:
    """Create random sparse column vector with dimension dim.

    Args:
        dim (int): vector dimension.

    Returns:
        sparse.coo_matrix: sparse column vector.
    """
    if dim <= 0:
        raise ValueError("dim must be positive.")
    rng = np.random.default_rng()

    nnz = max(1, int(0.1 * dim))
    rows = rng.choice(dim, size=nnz, replace=False)
    data = rng.normal(size=nnz)
    cols = np.zeros(nnz, dtype=int)
    return sparse.coo_matrix((data, (rows, cols)), shape=(dim, 1))


def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vector addition. 

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.

    Returns:
        np.ndarray: vector sum.
    """
    xc, yc = _as_col(x), _as_col(y)
    if xc.shape != yc.shape:
        raise ValueError(f"Shape mismatch: {xc.shape} vs {yc.shape}")
    return xc + yc


def scalar_multiplication(x: np.ndarray, a: float) -> np.ndarray:
    """Vector multiplication by scalar.

    Args:
        x (np.ndarray): vector.
        a (float): scalar.

    Returns:
        np.ndarray: multiplied vector.
    """
    return _as_col(x) * float(a)


def linear_combination(vectors: Sequence[np.ndarray], coeffs: Sequence[float]) -> np.ndarray:
    """Linear combination of vectors.

    Args:
        vectors (Sequence[np.ndarray]): list of vectors of len N.
        coeffs (Sequence[float]): list of coefficients of len N.

    Returns:
        np.ndarray: linear combination of vectors.
    """
    if len(vectors) != len(coeffs):
        raise ValueError("vectors and coeffs must have the same length.")
    if len(vectors) == 0:
        raise ValueError("vectors must be a non-empty sequence.")
    acc = np.zeros_like(_as_col(vectors[0]), dtype=float)
    for v, a in zip(vectors, coeffs):
        vc = _as_col(v)
        if vc.shape != acc.shape:
            raise ValueError("All vectors must have the same dimension/shape.")
        acc += float(a) * vc
    return acc


def dot_product(x: np.ndarray, y: np.ndarray) -> float:
    """Vectors dot product.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.

    Returns:
        float: dot product.
    """
    x1, y1 = _as_1d(x), _as_1d(y)
    if x1.shape != y1.shape:
        raise ValueError(f"Shape mismatch: {x1.shape} vs {y1.shape}")
    return float(np.dot(x1, y1))


def norm(x: np.ndarray, order: int | float) -> float:
    """Vector norm: Manhattan, Euclidean or Max.

    Args:
        x (np.ndarray): vector
        order (int | float): norm's order: 1, 2 or inf.

    Returns:
        float: vector norm
    """
    x1 = _as_1d(x)
    if order == 1:
        return float(np.linalg.norm(x1, 1))
    elif order == 2:
        return float(np.linalg.norm(x1, 2))
    elif order == np.inf or order == float("inf"):
        return float(np.linalg.norm(x1, np.inf))

    raise ValueError("Unsupported order. Use 1, 2, or np.inf.")


def distance(x: np.ndarray, y: np.ndarray) -> float:
    """L2 distance between vectors.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.

    Returns:
        float: distance.
    """
    x1, y1 = _as_1d(x), _as_1d(y)
    if x1.shape != y1.shape:
        raise ValueError(f"Shape mismatch: {x1.shape} vs {y1.shape}")
    return float(np.linalg.norm(x1 - y1))


def cos_between_vectors(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine between vectors in deg.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.

    Returns:
        np.ndarray: angle in deg.
    """
    x1, y1 = _as_1d(x), _as_1d(y)
    if x1.shape != y1.shape:
        raise ValueError(f"Shape mismatch: {x1.shape} vs {y1.shape}")
    nx = np.linalg.norm(x1)
    ny = np.linalg.norm(y1)
    if nx == 0.0 or ny == 0.0:
        raise ValueError("Angle is undefined for zero vector(s).")
    cos_th = float(np.dot(x1, y1) / (nx * ny))

    cos_th = max(-1.0, min(1.0, cos_th))
    angle_deg = np.degrees(np.arccos(cos_th))
    return float(angle_deg)


def is_orthogonal(x: np.ndarray, y: np.ndarray) -> bool:
    """Check is vectors orthogonal.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.

    Returns:
        bool: are vectors orthogonal.
    """
    x1, y1 = _as_1d(x), _as_1d(y)
    if x1.shape != y1.shape:
        raise ValueError(f"Shape mismatch: {x1.shape} vs {y1.shape}")

    if not np.any(x1) or not np.any(y1):
        return True
    dp = float(np.dot(x1, y1))
    nx = np.linalg.norm(x1)
    ny = np.linalg.norm(y1)

    tol = 1e-10 * nx * ny + 1e-12
    return abs(dp) <= tol


def solves_linear_systems(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve system of linear equations.

    Args:
        a (np.ndarray): coefficient matrix.
        b (np.ndarray): ordinate values.

    Returns:
        np.ndarray: sytems solution
    """
    A = np.asarray(a, dtype=float)
    b_arr = _as_1d(b)
    if A.ndim != 2:
        raise ValueError("a must be a 2D array.")
    n_rows, n_cols = A.shape
    if b_arr.shape[0] != n_rows:
        raise ValueError(f"Incompatible shapes: A is {A.shape}, b has length {b_arr.shape[0]}.")

    if n_rows == n_cols:
        x, *_ = np.linalg.lstsq(A, b_arr, rcond=None)
        return x.reshape(-1, 1)

    x, *_ = np.linalg.lstsq(A, b_arr, rcond=None)
    return x.reshape(-1, 1)
