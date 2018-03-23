import unittest

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from multistate_kernel.kernel import MultiStateKernel


class NumpyArrayAssertsTestCase(unittest.TestCase):
    def assertAllClose(self, first, second, tol=1e-5):
        self.assertTrue(np.allclose(first, second, rtol=tol), "Arrays aren't close:\n{}\n{}".format(first, second))


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

        call_result, call_grad = msk(X, None, True)
        expected_result = np.diag(np.r_[np.full(n1, level1), np.full(n2, level2)])
        expected_grad = np.zeros(shape=(n1+n2,n1+n2,3))
        expected_grad[np.arange(n1), np.arange(n1), 0] = np.full(n1, level1 * 2)
        expected_grad[n1+np.arange(n2), n1+np.arange(n2), 2] = np.full(n2, level2 * 2)
        expected_grad[np.arange(n1) + n1, np.arange(n1), 1] = np.full(n1, level1)
        expected_grad[np.arange(n1), np.arange(n1) + n1, 1] = np.full(n1, level1)
        self.assertAllClose(call_result, expected_result)
        self.assertAllClose(call_grad, expected_grad)
#        print(call_result)
#        print(np.swapaxes(call_grad,2,0))
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
        self.y2 = np.random.normal(size=sample_length)

        self.x = np.linspace(0.0, 1.0, num=sample_length).reshape(-1, 1)
        self.x = np.block([[np.zeros_like(self.x), self.x], [np.ones_like(self.x), self.x]])
        self.y = np.hstack((self.y1, self.y2))

    def test_multistate_kernel_for_independent_kernels(self):
        k1 = WhiteKernel(noise_level=1, noise_level_bounds='fixed')
        k2 = WhiteKernel(noise_level=1, noise_level_bounds='fixed')

        ms_kernel = MultiStateKernel((k1, k2,), np.array([[1,0],[-0.5,1]]), [np.array([[0.0,0.0],[0.0,0.0]]), np.array([[2.0,2.0],[2.0,2.0]])])
        gpr_msk = GaussianProcessRegressor(kernel=ms_kernel, random_state=0)
        gpr_msk.fit(self.x, self.y)
        self.assertAllClose(gpr_msk.kernel_.theta, np.array([1.0, 0.0, 1.0]), 1e-1)

class DependentDistributionsTestCase(NumpyArrayAssertsTestCase):
    def setUp(self):
        np.random.seed(42)

        sample_length = 1000
        self.y1 = np.random.normal(size=sample_length)
        self.y2 = self.y1 + np.random.normal(size=sample_length)

        self.x = np.linspace(0.0, 1.0, num=sample_length).reshape(-1, 1)
        self.x = np.block([[np.zeros_like(self.x), self.x], [np.ones_like(self.x), self.x]])
        self.y = np.hstack((self.y1, self.y2))

    def test_multistate_kernel_for_dependent_kernels(self):
        k1 = WhiteKernel(noise_level=1, noise_level_bounds='fixed')
        k2 = WhiteKernel(noise_level=1, noise_level_bounds='fixed')

        ms_kernel = MultiStateKernel((k1, k2,), np.array([[1,0],[-0.5,1]]), [np.array([[0.0,0.0],[0.0,0.0]]), np.array([[2.0,2.0],[2.0,2.0]])])
        gpr_msk = GaussianProcessRegressor(kernel=ms_kernel, random_state=0)
        gpr_msk.fit(self.x, self.y)
        self.assertAllClose(
            gpr_msk.kernel_.get_params()['scale'],
            np.array([[1.0, 0.0], [1.0,1.0]]),
            1e-1
        )
