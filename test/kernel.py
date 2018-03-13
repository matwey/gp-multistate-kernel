import unittest

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from multistate_kernel.kernel import MultiStateKernel


class NumpyArrayAssertsTestCase(unittest.TestCase):
    def assertAllClose(self, first, second):
        self.assertTrue(np.allclose(first, second), "Arrays aren't close:\n{}\n{}".format(first, second))


class MultiStateKernelCallTestCase(NumpyArrayAssertsTestCase):
    def test_two_white_noises_unity_matrix(self):
        level1 = 1
        level2 = 2
        matrix = np.eye(2)
        n1 = 3
        n2 = 5

        k1 = WhiteKernel(noise_level=level1, noise_level_bounds='fixed')
        k2 = WhiteKernel(noise_level=level2, noise_level_bounds='fixed')
        msk = MultiStateKernel((k1, k2,), matrix, [matrix]*2)
        X = np.block([[np.zeros(n1),  np.ones(n2)],
                      [np.arange(n1), np.arange(n2)]]).T

        call_result = msk(X)
        expected_result = np.diag(np.r_[np.full(n1, level1), np.full(n2, level2)])
        self.assertAllClose(call_result, expected_result)


class IndependentDistributionsTestCase(NumpyArrayAssertsTestCase):
    def setUp(self):
        self.x1 = self.x2 = np.arange(10).reshape(-1, 1)
        self.X = np.block([[np.zeros_like(self.x1), self.x1], [np.ones_like(self.x2), self.x2]])
        self.y1 = np.sin(self.x1).reshape(-1)
        self.y2 = np.power(self.x1, 2).reshape(-1)
        self.y = np.hstack((self.y1, self.y2))

        self.x1_ = np.linspace(self.x1.min(), self.x1.max(), 100).reshape(-1, 1)
        self.x2_ = np.linspace(self.x2.min(), self.x2.max(), 100).reshape(-1, 1)

    def test_multistate_kernel_for_independent_kernels(self):
        # TODO: fix RuntimeWarning through np.log from matrix bounds
        k1 = WhiteKernel(noise_level=1, noise_level_bounds='fixed')
        k2 = WhiteKernel(noise_level=1, noise_level_bounds='fixed')

        gpr1 = GaussianProcessRegressor(kernel=k1, random_state=0)
        gpr1.fit(self.x1, self.y1, )
        gpr2 = GaussianProcessRegressor(kernel=k2, random_state=0)
        gpr2.fit(self.x2, self.y2)
        ms_kernel = MultiStateKernel((k1, k2,), np.array([[1,0],[0,1]]), [np.array([[0.5,-1],[-1,0.5]]), np.array([[1.5,1],[1,1.5]])])
        gpr_msk = GaussianProcessRegressor(kernel=ms_kernel, random_state=0)
        gpr_msk.fit(self.X, self.y)

        y1_ = gpr1.predict(self.x1_)
        y2_ = gpr2.predict(self.x2_)
        y1_msk_ = gpr_msk.predict(np.hstack((np.zeros_like(self.x1_), self.x1_)))
        y2_msk_ = gpr_msk.predict(np.hstack((np.ones_like(self.x2_),  self.x2_)))

        self.assertAllClose(y1_, y1_msk_)
        self.assertAllClose(y2_, y2_msk_)
