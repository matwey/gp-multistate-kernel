import unittest

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from multistate_kernel.kernel import MultiStateKernel


class IndependentDistributionsTestCase(unittest.TestCase):
    def assertAllClose(self, first, second):
        self.assertTrue(np.allclose(first, second), "Arrays aren't close:\n{}\n{}".format(first, second))

    def test_multistate_kernel_for_independent_kernels(self):
        k1 = RBF(length_scale=1)
        k2 = RBF(length_scale=0.1)
        x1 = x2 = np.arange(10).reshape(-1,1)
        X = np.block([[np.zeros_like(x1), x1], [np.ones_like(x2), x2]])
        y1 = np.sin(x1).reshape(-1)
        y2 = np.power(x1, 2).reshape(-1)
        y = np.hstack((y1, y2))

        gpr1 = GaussianProcessRegressor(kernel=k1, random_state=0)
        gpr1.fit(x1, y1)
        gpr2 = GaussianProcessRegressor(kernel=k2, random_state=0)
        gpr2.fit(x2, y2)
        ms_kernel = MultiStateKernel((k1, k2,))
        gpr_msk = GaussianProcessRegressor(kernel=ms_kernel, random_state=0)
        gpr_msk.fit(X, y)

        x1_ = np.linspace(x1.min(), x1.max(), 100).reshape(-1,1)
        x2_ = np.linspace(x2.min(), x2.max(), 100).reshape(-1,1)
        y1_ = gpr1.predict(x1_)
        y2_ = gpr2.predict(x2_)
        y1_msk_ = gpr_msk.predict(np.hstack((np.zeros_like(x1_), x1_)))
        y2_msk_ = gpr_msk.predict(np.hstack((np.ones_like(x2_),  x2_)))

        self.assertAllClose(y1_, y1_msk_)
        self.assertAllClose(y2_, y2_msk_)
