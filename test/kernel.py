import unittest

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from multistate_kernel.kernel import MultiStateKernel, ConstantMatrix


class IndependentDistributionsTestCase(unittest.TestCase):
    def setUp(self):
        self.x1 = self.x2 = np.arange(10).reshape(-1, 1)
        self.X = np.block([[np.zeros_like(self.x1), self.x1], [np.ones_like(self.x2), self.x2]])
        self.y1 = np.sin(self.x1).reshape(-1)
        self.y2 = np.power(self.x1, 2).reshape(-1)
        self.y = np.hstack((self.y1, self.y2))

        self.x1_ = np.linspace(self.x1.min(), self.x1.max(), 100).reshape(-1, 1)
        self.x2_ = np.linspace(self.x2.min(), self.x2.max(), 100).reshape(-1, 1)

    def assertAllClose(self, first, second):
        self.assertTrue(np.allclose(first, second), "Arrays aren't close:\n{}\n{}".format(first, second))

    def test_constant_matrix_with_diagonal_form(self):
        c1 = 1
        c2 = 2
        bounds = [1e-3, 1e3]

        k1 = ConstantKernel(constant_value=c1, constant_value_bounds=bounds)
        k2 = ConstantKernel(constant_value=c2, constant_value_bounds=bounds)
        cmk = ConstantMatrix(np.array([c1, c2]), (np.array([bounds[0]]*2), np.array([bounds[1]]*2)))

        gpr1 = GaussianProcessRegressor(kernel=k1, random_state=0)
        gpr1.fit(self.x1, self.y1, )
        gpr2 = GaussianProcessRegressor(kernel=k2, random_state=0)
        gpr2.fit(self.x2, self.y2)
        gpr_cmk = GaussianProcessRegressor(kernel=cmk, random_state=0)
        gpr_cmk.fit(self.X, self.y)

    def test_multistate_kernel_for_independent_kernels(self):
        k1 = RBF(length_scale=1)
        k2 = RBF(length_scale=0.1)

        gpr1 = GaussianProcessRegressor(kernel=k1, random_state=0)
        gpr1.fit(self.x1, self.y1, )
        gpr2 = GaussianProcessRegressor(kernel=k2, random_state=0)
        gpr2.fit(self.x2, self.y2)
        ms_kernel = MultiStateKernel((k1, k2,))
        gpr_msk = GaussianProcessRegressor(kernel=ms_kernel, random_state=0)
        gpr_msk.fit(self.X, self.y)

        y1_ = gpr1.predict(self.x1_)
        y2_ = gpr2.predict(self.x2_)
        y1_msk_ = gpr_msk.predict(np.hstack((np.zeros_like(self.x1_), self.x1_)))
        y2_msk_ = gpr_msk.predict(np.hstack((np.ones_like(self.x2_),  self.x2_)))

        self.assertAllClose(y1_, y1_msk_)
        self.assertAllClose(y2_, y2_msk_)
