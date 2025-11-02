import numpy as np

def negative_matrix(x: np.ndarray) -> np.ndarray:
    """
    Returns the negation of each element in the input vector or matrix.

    Args:
        x (np.ndarray): A vector (n*1) or matrix (n*n).

    Returns:
        np.ndarray: A matrix with each element negated.
    """
    x = np.asarray(x)
    if x.ndim not in (1, 2):
        raise ValueError("x must be a 1D vector or 2D matrix.")
    return -x


def reverse_matrix(x: np.ndarray) -> np.ndarray:
    """
    Returns the input vector or matrix with the order of elements reversed.

    Args:
        x (np.ndarray): A vector (n*1) or matrix (n*n).

    Returns:
        np.ndarray: A matrix with the order of elements reversed.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return x[::-1]
    if x.ndim == 2:
        return x[::-1, ::-1]
    raise ValueError("x must be a 1D vector or 2D matrix.")


def affine_transform(
    x: np.ndarray, alpha_deg: float, scale: tuple[float, float], shear: tuple[float, float],
    translate: tuple[float, float],
) -> np.ndarray:
    """Compute affine transformation

    Args:
        x (np.ndarray): vector n*1 or matrix n*n.
        alpha_deg (float): rotation angle in deg.
        scale (tuple[float, float]): x, y scale factor.
        shear (tuple[float, float]): x, y shear factor.
        translate (tuple[float, float]): x, y translation factor.

    Returns:
        np.ndarray: transformed matrix.
    """
    pts = np.asarray(x, dtype=float)

    transposed_back = False
    if pts.ndim != 2:
        raise ValueError("x must be a 2D array of points with shape (2, m) or (m, 2).")

    if pts.shape[0] == 2:
        P = pts
    elif pts.shape[1] == 2:
        P = pts.T
        transposed_back = True
    else:
        raise ValueError("x must have shape (2, m) or (m, 2) representing 2D points.")

    alpha = np.deg2rad(float(alpha_deg))
    c, s = np.cos(alpha), np.sin(alpha)
    sx, sy = map(float, scale)
    kx, ky = map(float, shear)
    tx, ty = map(float, translate)

    R = np.array([[ c, -s, 0.0],
                  [ s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=float)
    S = np.array([[sx, 0.0, 0.0],
                  [0.0, sy, 0.0],
                  [0.0, 0.0, 1.0]], dtype=float)
    Sh = np.array([[1.0, kx, 0.0],
                   [ky, 1.0, 0.0],
                   [0.0, 0.0, 1.0]], dtype=float)
    T = np.array([[1.0, 0.0, tx],
                  [0.0, 1.0, ty],
                  [0.0, 0.0, 1.0]], dtype=float)

    A = T @ Sh @ S @ R

    ones = np.ones((1, P.shape[1]), dtype=float)
    Ph = np.vstack([P, ones])
    Pp = A @ Ph
    out = Pp[:2, :]

    return out.T if transposed_back else out
