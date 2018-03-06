import unittest

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from multistate_kernel.kernel import MultiStateKernel


class IndependentDistributionsTestCase(unittest.TestCase):
    def test_multistate_kernel_for_independent_kernels(self):
        k1 = RBF(length_scale=1)
        k2 = RBF(length_scale=10)
        x1 = x2 = np.arange(10).reshape(-1,1)
        X = np.block([[np.zeros_like(x1), x1], [np.ones_like(x2), x2]])
        y1 = np.sin(x1).reshape(-1)
        y2 = np.cos(x1).reshape(-1)**2
        y = np.hstack((y1, y2))

        ms_kernel = MultiStateKernel((k1, k2,))
        gpr_msk = GaussianProcessRegressor(kernel=ms_kernel)
        gpr_msk.fit(X, y)