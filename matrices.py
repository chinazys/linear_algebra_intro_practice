import numpy as np
from scipy.linalg import qr as scipy_qr


def get_matrix(n: int, m: int) -> np.ndarray:
    """Create random matrix n * m.

    Args:
        n (int): number of rows.
        m (int): number of columns.

    Returns:
        np.ndarray: matrix n*m.
    """
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive integers.")
    rng = np.random.default_rng()
    return rng.normal(size=(n, m))


def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Matrix addition.

    Args:
        x (np.ndarray): 1st matrix.
        y (np.ndarray): 2nd matrix.

    Returns:
        np.ndarray: matrix sum.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
    return x + y


def scalar_multiplication(x: np.ndarray, a: float) -> np.ndarray:
    """Matrix multiplication by scalar.

    Args:
        x (np.ndarray): matrix.
        a (float): scalar.

    Returns:
        np.ndarray: multiplied matrix.
    """
    x = np.asarray(x, dtype=float)
    return x * a


def dot_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Matrices dot product.

    Args:
        x (np.ndarray): 1st matrix.
        y (np.ndarray): 2nd matrix or vector.

    Returns:
        np.ndarray: dot product.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be 2D.")
    if y.ndim not in (1, 2):
        raise ValueError("y must be 1D or 2D.")
    return x @ y


def identity_matrix(dim: int) -> np.ndarray:
    """Create identity matrix with dimension `dim`. 

    Args:
        dim (int): matrix dimension.

    Returns:
        np.ndarray: identity matrix.
    """
    if dim <= 0:
        raise ValueError("dim must be positive.")
    return np.eye(dim, dtype=float)


def matrix_inverse(x: np.ndarray) -> np.ndarray:
    """Compute inverse matrix.

    Args:
        x (np.ndarray): matrix.

    Returns:
        np.ndarray: inverse matrix.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError("x must be a square 2D matrix to invert.")
    return np.linalg.inv(x)
    

def matrix_transpose(x: np.ndarray) -> np.ndarray:
    """Compute transpose matrix.

    Args:
        x (np.ndarray): matrix.

    Returns:
        np.ndarray: transosed matrix.
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("x must be a 2D matrix.")
    return x.T


def hadamard_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute hadamard product.

    Args:
        x (np.ndarray): 1th matrix.
        y (np.ndarray): 2nd matrix.

    Returns:
        np.ndarray: hadamard produc
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
    return x * y


def basis(x: np.ndarray) -> tuple[int]:
    """Compute matrix basis.

    Args:
        x (np.ndarray): matrix.

    Returns:
        tuple[int]: indexes of basis columns.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be a 2D matrix.")
    n, m = x.shape
    if n == 0 or m == 0:
        return tuple()

    Q, R, piv = scipy_qr(x, pivoting=True, mode="economic")
    if R.size == 0:
        r = 0
    else:
        diag = np.abs(np.diag(R))
        tol = np.finfo(float).eps * max(x.shape) * (diag.max() if diag.size else 0.0)
        r = int(np.sum(diag > tol))

    idx = tuple(sorted(map(int, piv[:r])))
    return idx


def norm(x: np.ndarray, order: int | float | str) -> float:
    """Matrix norm: Frobenius, Spectral or Max.

    Args:
        x (np.ndarray): vector
        order (int | float): norm's order: 'fro', 2 or inf.

    Returns:
        float: vector norm
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be a 2D matrix.")
    return float(np.linalg.norm(x, ord=order))