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
    def test_two_white_noises_unity_matrix_correlated(self):
        level1 = 1
        level2 = 2
        inter = 0.5
        matrix = np.array([[1,0],[inter,1]])
        n1 = 3
        n2 = 5

        k1 = WhiteKernel(noise_level=level1, noise_level_bounds='fixed')
        k2 = WhiteKernel(noise_level=level2, noise_level_bounds='fixed')
        msk = MultiStateKernel((k1, k2,), matrix, [matrix]*2)
        X = np.block([[np.zeros(n1),  np.ones(n2)],
                      [np.arange(n1), np.arange(n2)]]).T

        call_result = msk(X)
        expected_result = np.diag(np.r_[np.full(n1, level1), np.full(n2, level2 + inter**2)])
        expected_result[np.arange(n1) + n1, np.arange(n1)] = np.full(n1, inter * level1)
        expected_result[np.arange(n1), np.arange(n1) + n1] = np.full(n1, inter * level1)
        self.assertAllClose(call_result, expected_result)

class IndependentDistributionsTestCase(NumpyArrayAssertsTestCase):
    def setUp(self):
        np.random.seed(42)

        sample_length = 1000
        self.y1 = np.random.normal(size=sample_length)
        self.y2 = self.y1 + np.random.normal(size=sample_length)

        self.x = np.linspace(0.0, 1.0, num=sample_length).reshape(-1, 1)
        self.x = np.block([[np.zeros_like(self.x), self.x], [np.ones_like(self.x), self.x]])
        self.y = np.hstack((self.y1, self.y2))

    def test_multistate_kernel_for_independent_kernels(self):
        k1 = WhiteKernel(noise_level=1, noise_level_bounds='fixed')
        k2 = WhiteKernel(noise_level=1, noise_level_bounds='fixed')

        ms_kernel = MultiStateKernel((k1, k2,), np.array([[1,0],[0.5,1]]), [np.array([[0.5,-1],[-1,0.5]]), np.array([[1.5,1],[1,1.5]])])
        gpr_msk = GaussianProcessRegressor(kernel=ms_kernel, random_state=0)
        gpr_msk.fit(self.x, self.y)
