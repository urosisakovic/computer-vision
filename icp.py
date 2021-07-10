from typing import Tuple

import numpy as np


def rigid_aligment(
        a: np.ndarray,
        b: np.ndarray,
        w: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds translation and rotation such that best align point cloud a with point cloud b
    given that each point a[i] corresponds to the point b[i] (correspondences are known).

    Args:
        a: A point cloud of shape [<number of points>, 3].
        b: A point cloud of shape [<number of points>, 3].
        w: Weights with shape [<number of points>,]. w[i] denotes certainty (or importance)
            of the point pair a[i] and b[i]. If none, all pairs will be treated equally.
            If provided, sum of all weights must be equal to 1.
    Rets:
        R(3x3) and t(3x1), NumPy arrays denoting rotation and translation leading to the optimal alignment
        of two given point clouds.
    """
    assert len(a.shape) == 2
    assert a.shape[1] == 3
    assert a.shape == b.shape
    if w is not None:
        assert len(w.shape) == 1
        assert w.shape[0] == a.shape[0]
        assert np.isclose(np.sum(w), 1.0)

    num_of_points = a.shape[0]

    # If weights are not provided, all pairs have the same importance
    if w is None:
        w = np.ones((num_of_points,)) / num_of_points

    w = np.expand_dims(w, axis=-1)

    a_weighted_mean = np.sum(a * w, axis=0).reshape(3)
    b_weighted_mean = np.sum(b * w, axis=0).reshape(3)

    H = np.zeros((3, 3))

    # TODO(urosisakovic): Vectorize this
    for (a_point, b_point, weight) in zip(a, b, w):
        H += (a_point - a_weighted_mean).reshape(3, 1) * (b_point - b_weighted_mean).reshape(1, 3) * weight

    print(H.shape)

    u, _, vh = np.linalg.svd(H)

    R = vh @ u.T

    t = b_weighted_mean.reshape(3, 1) - R @ a_weighted_mean.reshape(3, 1)

    assert R.shape == (3, 3)
    assert t.shape == (3, 1)

    return R, t
    

def icp():
    pass