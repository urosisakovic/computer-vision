import numpy as np
from scipy.spatial.transform import Rotation
from icp import rigid_aligment


def test_rigid_aligment():
    # Fix seed in order to make test reproducible
    np.random.seed(0)
    
    N = 1000

    a = np.random.uniform(size=(N, 3))
    b = a.copy()

    R_true = Rotation.from_rotvec(np.pi/2 * np.array([0.25, 0.5, 1])).as_matrix()
    t_true = np.array([10, 20, 30]).reshape(3, 1)

    b = R_true @ b.T + t_true
    b = b.T

    R_est, t_est = rigid_aligment(a, b)

    assert np.allclose(t_est, t_true)
    assert np.allclose(R_est, R_true)
