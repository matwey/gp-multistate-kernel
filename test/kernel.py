import unittest

import math
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern

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

class IndependentWhiteKernelTestCase(NumpyArrayAssertsTestCase):
    def setUp(self):
        np.random.seed(42)

        sample_length = 1000
        self.y1 = np.random.normal(size=sample_length)
        self.y2 = np.random.normal(size=sample_length)

        self.x = np.linspace(0.0, 1.0, num=sample_length).reshape(-1, 1)
        self.x = np.block([[np.zeros_like(self.x), self.x], [np.ones_like(self.x), self.x]])
        self.y = np.hstack((self.y1, self.y2))

    def test_multistate_kernel_for_independent_white_kernels(self):
        k1 = WhiteKernel(noise_level=1, noise_level_bounds='fixed')
        k2 = WhiteKernel(noise_level=1, noise_level_bounds='fixed')

        ms_kernel = MultiStateKernel((k1, k2,), np.array([[1,0],[-0.5,1]]), [np.array([[0.0,0.0],[0.0,0.0]]), np.array([[2.0,2.0],[2.0,2.0]])])
        gpr_msk = GaussianProcessRegressor(kernel=ms_kernel, random_state=0)
        gpr_msk.fit(self.x, self.y)
        self.assertAllClose(gpr_msk.kernel_.theta, np.array([1.0, 0.0, 1.0]), 1e-1)

class IndependentARTestCase(NumpyArrayAssertsTestCase):
    def setUp(self):
        np.random.seed(42)

        ar = 0.5
        sample_length = 1000
        self.y1 = np.random.normal(size=sample_length)
        self.y2 = np.random.normal(size=sample_length)

        for i in range(2, sample_length):
                self.y1[i] = ar * self.y1[i-1] + self.y1[i]
                self.y2[i] = ar * self.y2[i-1] + self.y2[i]

        self.x1 = np.linspace(0.0, 1.0 * sample_length, num=sample_length).reshape(-1, 1)
        self.x = np.block([[np.zeros_like(self.x1), self.x1], [np.ones_like(self.x1), self.x1]])
        self.y = np.hstack((self.y1, self.y2))

    def test_multistate_kernel_for_independent_ar_kernels(self):
        k1 = Matern(nu=0.5, length_scale=1.0, length_scale_bounds=(0.01, 100))
        k2 = Matern(nu=0.5, length_scale=1.0, length_scale_bounds=(0.01, 100))

        gpr_k1 = GaussianProcessRegressor(kernel=1.0*k1, random_state=0)
        gpr_k2 = GaussianProcessRegressor(kernel=1.0*k2, random_state=0)

        gpr_k1.fit(self.x1, self.y1)
        gpr_k2.fit(self.x1, self.y2)

        expected_length_scale = -1.0/math.log(0.5)
        expected_constant = 1.25
        params_k1 = gpr_k1.kernel_.get_params()
        params_k2 = gpr_k2.kernel_.get_params()
        self.assertAlmostEqual(params_k1['k2__length_scale'], expected_length_scale, delta=0.1)
        self.assertAlmostEqual(params_k2['k2__length_scale'], expected_length_scale, delta=0.1)
        self.assertAlmostEqual(params_k1['k1__constant_value'], expected_constant, delta=0.1)
        self.assertAlmostEqual(params_k2['k1__constant_value'], expected_constant, delta=0.1)

        ms_kernel = MultiStateKernel((k1, k2,), np.array([[1,0],[-0.5,1]]), [np.array([[0.0,0.0],[0.0,0.0]]), np.array([[2.0,2.0],[2.0,2.0]])])
        gpr_msk = GaussianProcessRegressor(kernel=ms_kernel, random_state=0)
        gpr_msk.fit(self.x, self.y)

        params_msk=gpr_msk.kernel_.get_params()
        self.assertAlmostEqual(params_msk['s0__length_scale'], expected_length_scale, delta=0.1)
        self.assertAlmostEqual(params_msk['s1__length_scale'], expected_length_scale, delta=0.1)
        self.assertAllClose(
            params_msk['scale'],
            np.array([[expected_constant**0.5, 0.0], [0.0,expected_constant**0.5]]),
            1e-1
        )

class DependentWhiteKernelTestCase(NumpyArrayAssertsTestCase):
    def setUp(self):
        np.random.seed(42)

        sample_length = 1000
        self.y1 = np.random.normal(size=sample_length)
        self.y2 = self.y1 + np.random.normal(size=sample_length)

        self.x = np.linspace(0.0, 1.0, num=sample_length).reshape(-1, 1)
        self.x = np.block([[np.zeros_like(self.x), self.x], [np.ones_like(self.x), self.x]])
        self.y = np.hstack((self.y1, self.y2))

    def test_multistate_kernel_for_dependent_white_kernels(self):
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

class DependentARTestCase(NumpyArrayAssertsTestCase):
    def setUp(self):
        np.random.seed(42)

        ar = 0.5
        sample_length = 1000
        self.y1 = np.random.normal(size=sample_length)
        self.y2 = np.random.normal(size=sample_length)

        for i in range(2, sample_length):
                self.y1[i] = ar * self.y1[i-1] + self.y1[i]
                self.y2[i] = ar * self.y2[i-1] + self.y2[i]

        self.x1 = np.linspace(0.0, 1.0 * sample_length, num=sample_length).reshape(-1, 1)
        self.x = np.block([[np.zeros_like(self.x1), self.x1], [np.ones_like(self.x1), self.x1]])
        self.y = np.hstack((self.y1, self.y2 - self.y1))

    def test_multistate_kernel_for_dependent_ar_kernels(self):
        k1 = Matern(nu=0.5, length_scale=1.0, length_scale_bounds=(0.01, 100))
        k2 = Matern(nu=0.5, length_scale=1.0, length_scale_bounds=(0.01, 100))

        gpr_k1 = GaussianProcessRegressor(kernel=1.0*k1, random_state=0)
        gpr_k2 = GaussianProcessRegressor(kernel=1.0*k2, random_state=0)

        gpr_k1.fit(self.x1, self.y1)
        gpr_k2.fit(self.x1, self.y2)

        expected_length_scale = -1.0/math.log(0.5)
        expected_constant = 1.25
        params_k1 = gpr_k1.kernel_.get_params()
        params_k2 = gpr_k2.kernel_.get_params()
        self.assertAlmostEqual(params_k1['k2__length_scale'], expected_length_scale, delta=0.1)
        self.assertAlmostEqual(params_k2['k2__length_scale'], expected_length_scale, delta=0.1)
        self.assertAlmostEqual(params_k1['k1__constant_value'], expected_constant, delta=0.1)
        self.assertAlmostEqual(params_k2['k1__constant_value'], expected_constant, delta=0.1)

        ms_kernel = MultiStateKernel((k1, k2,), np.array([[1,0],[-0.5,1]]), [np.array([[-2.0,-2.0],[-2.0,-2.0]]), np.array([[2.0,2.0],[2.0,2.0]])])
        gpr_msk = GaussianProcessRegressor(kernel=ms_kernel, random_state=0)
        gpr_msk.fit(self.x, self.y)

        params_msk=gpr_msk.kernel_.get_params()
        self.assertAlmostEqual(params_msk['s0__length_scale'], expected_length_scale, delta=0.1)
        self.assertAlmostEqual(params_msk['s1__length_scale'], expected_length_scale, delta=0.1)
        self.assertAllClose(
            params_msk['scale'],
            np.array([[expected_constant**0.5, 0.0], [-expected_constant**0.5,expected_constant**0.5]]),
            1e-1
        )

class DependentMixedTestCase(NumpyArrayAssertsTestCase):
    def setUp(self):
        np.random.seed(42)

        ar = 0.5
        self.sample_length = 1000
        self.y1 = np.random.normal(size=self.sample_length)
        self.y2 = 1.25**0.5 * np.random.normal(size=self.sample_length)

        for i in range(2, self.sample_length):
                self.y1[i] = ar * self.y1[i-1] + self.y1[i]

        self.x1 = np.linspace(0.0, 1.0 * self.sample_length, num=self.sample_length).reshape(-1, 1)
        self.x = np.block([[np.zeros_like(self.x1), self.x1], [np.ones_like(self.x1), self.x1]])
        self.y = np.hstack((self.y1, self.y2 - self.y1))

    def test_multistate_sample_for_mixed_kernels(self):
        k1 = Matern(nu=0.5, length_scale=1.0, length_scale_bounds=(0.01, 100))
        k2 = WhiteKernel(noise_level=1, noise_level_bounds='fixed')
        ms_kernel = MultiStateKernel((k1, k2,), np.array([[1,0],[-0.5,1]]), [np.array([[-2.0,-2.0],[-2.0,-2.0]]), np.array([[2.0,2.0],[2.0,2.0]])])
        gpr_msk = GaussianProcessRegressor(kernel=ms_kernel, random_state=0)
        gpr_msk.fit(self.x, self.y)
        y_samples = gpr_msk.sample_y(self.x, random_state=0)
        gpr_msk2 = GaussianProcessRegressor(kernel=ms_kernel, random_state=0)
        gpr_msk2.fit(self.x, y_samples.reshape(-1))
        self.assertAllClose(gpr_msk.kernel_.theta, gpr_msk2.kernel_.theta, 1e-1)

    def test_multistate_predict_for_mixed_kernels(self):
        k1 = Matern(nu=0.5, length_scale=1.0, length_scale_bounds=(0.01, 100))
        k2 = WhiteKernel(noise_level=1, noise_level_bounds='fixed')
        ms_kernel = MultiStateKernel((k1, k2,), np.array([[1,0],[-0.5,1]]), [np.array([[-2.0,-2.0],[-2.0,-2.0]]), np.array([[2.0,2.0],[2.0,2.0]])])
        gpr_msk = GaussianProcessRegressor(kernel=ms_kernel, random_state=0)
        gpr_msk.fit(self.x, self.y)
        x1 = np.linspace(0.25 * self.sample_length, 0.75 * self.sample_length, num=self.sample_length).reshape(-1, 1)
        x = np.block([[np.zeros_like(x1), x1], [np.ones_like(x1), x1]])
        mean = gpr_msk.predict(x)
        mean1 = mean[:self.sample_length]
        mean2 = mean[self.sample_length:]
        self.assertAllClose(mean1, -mean2, 1e-1)

    def test_multistate_kernel_for_mixed_kernels(self):
        k1 = Matern(nu=0.5, length_scale=1.0, length_scale_bounds=(0.01, 100))
        k2 = WhiteKernel(noise_level=1, noise_level_bounds='fixed')

        gpr_k1 = GaussianProcessRegressor(kernel=1.0*k1, random_state=0)

        gpr_k1.fit(self.x1, self.y1)

        expected_length_scale = -1.0/math.log(0.5)
        expected_constant = 1.25
        params_k1 = gpr_k1.kernel_.get_params()
        self.assertAlmostEqual(params_k1['k2__length_scale'], expected_length_scale, delta=0.1)
        self.assertAlmostEqual(params_k1['k1__constant_value'], expected_constant, delta=0.1)

        ms_kernel = MultiStateKernel((k1, k2,), np.array([[1,0],[-0.5,1]]), [np.array([[-2.0,-2.0],[-2.0,-2.0]]), np.array([[2.0,2.0],[2.0,2.0]])])
        gpr_msk = GaussianProcessRegressor(kernel=ms_kernel, random_state=0)
        gpr_msk.fit(self.x, self.y)

        params_msk=gpr_msk.kernel_.get_params()
        self.assertAlmostEqual(params_msk['s0__length_scale'], expected_length_scale, delta=0.1)
        self.assertAllClose(
            params_msk['scale'],
            np.array([[expected_constant**0.5, 0.0], [-expected_constant**0.5, expected_constant**0.5]]),
            1e-1
        )
