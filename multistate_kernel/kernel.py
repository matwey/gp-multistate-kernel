import numpy as np
from collections import OrderedDict
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter

class VariadicKernelOperator(Kernel):
	"""Alternative to sklearn.gaussian_process.kernels.KernelOperator with variadic number nested kernel"""

	def __init__(self, **kernels):
		self.kernels = OrderedDict(sorted(kernels.items(), key=lambda t: t[0]))

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

		params = dict(self.kernels)
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
	def _get_kernel_dict(self):
		return dict([("s" + str(n), kernel) for n, kernel in enumerate(self.state_kernels)])

	def __init__(self, kernels):
		self.state_kernels = kernels
		kwargs = self._get_kernel_dict()
		super(MultiStateKernel, self).__init__(**kwargs)

	def get_params(self, deep=True):
		params = super(MultiStateKernel, self).get_params(self, deep)
		for name, kernel in self._get_kernel_dict().items():
			del params[name]
		params['kernels'] = self.state_kernels
		return params

	def __call__(self, X, Y=None, eval_gradient=False):
		"""Return the kernel k(X, Y) and optionally its gradient.

		Parameters
		----------
		X : array, shape (n_samples_X, n_features)
			Left argument of the returned kernel k(X, Y)
		Y : array, shape (n_samples_Y, n_features), (optional, default=None)
			Right argument of the returned kernel k(X, Y). If None, k(X, X)
			if evaluated instead.
		eval_gradient : bool (optional, default=False)
			Determines whether the gradient with respect to the kernel
			hyperparameter is determined. Only supported when Y is None.

		Returns
		-------
		K : array, shape (n_samples_X, n_samples_Y)
			Kernel k(X, Y)
		K_gradient : array (opt.), shape (n_samples_X, n_samples_Y, n_dims)
			The gradient of the kernel k(X, X) with respect to the
			hyperparameter of the kernel. Only returned when eval_gradient
			is True.
		"""

		if Y is None:
			Y = X
		else:
			if eval_gradient:
				raise ValueError("Gradient can only be evaluated when Y is None.")

		n_samples_X, n_featuresX = X.shape
		n_samples_Y, n_featuresY = Y.shape

		assert n_featuresX == n_featuresY

		n_features = n_featuresX

		K = np.zeros(shape=(n_samples_X, n_samples_Y))
		if eval_gradient:
			K_gradient = np.zeros(shape=(n_samples_X, n_samples_Y, self.n_dims))

		states_X = X[:,0].astype(int)
		states_Y = Y[:,0].astype(int)

		pos = 0
		for n, kernel in enumerate(self.state_kernels):
			related_X = (states_X == n)
			related_Y = (states_Y == n)

			if eval_gradient:
				n = pos + kernel.n_dims
				K[np.ix_(related_X,related_Y)], K_gradient[np.ix_(related_X,related_Y,range(pos,n))] = kernel(X[related_X,1:], None, True)
				pos = n
			else:
				K[np.ix_(related_X,related_Y)] = kernel(X[related_X,1:], Y[related_Y,1:], False)

		if eval_gradient:
			return K, K_gradient
		return K

	def diag(self, X):
		"""Returns the diagonal of the kernel k(X, X).
		The result of this method is identical to np.diag(self(X)); however,
		it can be evaluated more efficiently since only the diagonal is
		evaluated.

		Parameters
		----------
			X : array, shape (n_samples_X, n_features)
				Left argument of the returned kernel k(X, Y)

		Returns
		-------
		K_diag : array, shape (n_samples_X,)
			Diagonal of kernel k(X, X)
		"""

		n_samples_X, n_features = X.shape

		K_diag = np.zeros(shape=(n_samples_X,))

		states_X = X[:,0].astype(int)

		for n, kernel in enumerate(self.state_kernels):
			related_X = (states_X == n)

			K_diag[related_X] = kernel.diag(X[related_X,1:])

		return K_diag
