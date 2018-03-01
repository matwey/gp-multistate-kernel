import numpy as np
from sklearn.gaussian_process.kernels import Kernel

class VariadicKernelOperator(Kernel):
	"""Alternative to sklearn.gaussian_process.kernels.KernelOperator with variadic number nested kernel"""

	def __init__(self, **kernels):
		self.kernels = kernels

	def get_params(self, deep=True):
		"""Get parameters of this kernel.
		Parameters
		----------
			deep : boolean, optional
				If True, will return the parameters for this estimator and
				contained subobjects that are estimators.
		Returns
		-------
			params : mapping of string to any
				Parameter names mapped to their values.
		"""

		params = self.kernels.copy()
		if deep:
			for name, kernel in self.kernels.items():
				deep_items = kernel.get_params().items()
				params.update((name + '__' + k, val) for k, val in deep_items)
		return params

	@property
	def hyperparameters(self):
		"""Returns a list of all hyperparameter."""

		r = []
		for name, kernel in self.kernels.items():
			for hyperparameter in kernel.hyperparameters:
				r.append(Hyperparameter(name + "__" + hyperparameter.name,
					hyperparameter.value_type,
					hyperparameter.bounds,
					hyperparameter.n_elements))
		return r

	@property
	def theta(self):
		return np.concatenate([kernel.theta for kernel in self.kernels.values()])

	@theta.setter
	def theta(self, theta):
		"""Sets the (flattened, log-transformed) non-fixed hyperparameters.
		Parameters
		----------
			theta : array, shape (n_dims,)
				The non-fixed, log-transformed hyperparameters of the kernel
		"""
		pos = 0
		for kernel in self.kernels.values():
			n = pos + kernel.n_dims
			kernel.theta = theta[pos:n]
			pos = n

	@property
	def bounds(self):
		return np.vstack([kernel.bounds for kernel in self.kernels.values() if kernel.bounds.size != 0])

	def __eq__(self, b):
		if type(self) != type(b):
			return False
		return self.kernels == b.kernels

	def is_stationary(self):
		return all([kernel.is_stationary() for kernel in self.kernels.values()])

class MultiStateKernel(VariadicKernelOperator):
	def __init__(self, *kernels):
		kwargs = dict([("s" + str(n), kernel) for n, kernel in enumerate(kernels)])
		super(MultiStateKernel, self).__init__(**kwargs)

	def __call__(self, X, Y=None, eval_gradient=False):
		super(MultiStateKernel, self).__call__(X,Y,eval_gradient)

	def diag(self, X):
		super(MultiStateKernel, self).diag(X)
