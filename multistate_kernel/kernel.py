import numpy as np
import math
from collections import OrderedDict
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter

class VariadicKernelOperator(Kernel):
	"""Variadic number nested kernel container.

	This is a base class to simplify handling multiple nested kernels as
	single one kernel. The class arranges parameters to be optimized and
	handles other simple basic stuff. The user should inherite from this
	class and implement the rest
	``sklearn.gaussian_process.kernels.KernelOperator`` interface, for
	instance, ``__call__()``.

	Parameters
	----------
	kernels : dict of sklearn.gaussian_process.kernels.Kernel
		The named collection of the kernels objects to be nested.

	See Also
	--------
	sklearn.gaussian_process.kernels.Kernel : Base kernel interface.
	sklearn.gaussian_process.kernels.KernelOperator : Kernel operator for two nested kernels.
	"""

	def __init__(self, **kernels):
		self.kernels = OrderedDict(sorted(kernels.items(), key=lambda t: t[0]))

	def get_params(self, deep=True):
		"""Get parameters of this kernel.

		Parameters
		----------
			deep : boolean, optional
				If True, will return the parameters for this
				estimator and contained subobjects that are estimators.
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
	"""Kernel for multi-state process

	This kernel handles multidimensional stochastic processes analysis
	inside ``sklearn.gaussian_process`` framework.

	Let us consider ``n_states`` different gaussian processes. Every process is
	allowed to have its own internal correlation properties. All processes in
	common are independent from each other. We call this processes as
	generating processes. At any time moment (X position) we can construct
	the vector :math:`{\\bf e}` having ``n_states`` elements where i-th element
	containts the value of i-th process. Since the generating processes are
	independent, covariance matrix for the generating vector is diagonal
	one.

	To introduce correlations between processes, the lower triangular
	scaling matrix :math:`{\\rm R}` is used. At every time moment we call
	:math:`{\\rm R} {\\bf e}` as observable vector and corresponding elements of
	vector :math:`{\\rm R} {\\bf e}` as observed processes.

	The user specify kernels for each generating process in natural order.
	The kernel parameters and scaling matrix :math:`{\\rm R}` are to be found by
	optimizer when training on real data. The user should not introduce
	multiplication constants in front of provided kernels. The scales are
	alread handled by the scaling matrix :math:`{\\rm R}` and additional
	parameters will make optimization problem degenerated.

	We assume that at every time-moment (X-position) only one observable
	process is measured. Since observable processes are correlated,
	measured value `Y` for one process leads to additional information in
	form of conditional probability for other processes at the same
	time-moment (X-position).

	The data for training should be provided in the following format.
	Each `X` sample must be prepended with additional element enumerating the
	observed process which the sample belogs to. This additional integer
	value is called a state. Corresponding `Y` contains measured value.
	It is supported to have two samples with different state but the same
	time-moment (X-position).

	For instance if we have two observable processes and measured the first
	process as the following

	+---+---+---+---+
	| X | 1 | 3 | 5 |
	+---+---+---+---+
	| Y |0.0|0.3|0.1|
	+---+---+---+---+

	The second process as the following

	+---+---+---+
	| X | 2 | 6 |
	+---+---+---+
	| Y |0.2|0.4|
	+---+---+---+


	Then expected input `X` for ``MultiStateKernel`` should be the
	following ``[[0,1],[1,2],[0,3],[0,5],[1,6]]``. And values `Y` are
	arranged with respect to `X`: ``[0.0, 0.2, 0.3, 0.1, 0.4]``.
	This is common rule for training and predict operations.

	Parameters
	----------
	kernels : list of ``sklearn.gaussian_process.kernels.Kernel``
		Array of kernels for each state. The array length should be
		n_states.
	scale : array, shape (n_states, n_states)
		Initial lower triangular scale matrix.
	scale_bounds : array, shape (2, n_states, n_states)
		Lower and upper bounds for elements of the scale matrix.

	Examples
	--------
	Here, we construct ``MultiStateKernel`` for the case of two states.
	The first generating process is expected to obey Matern kernel. The
	second one is white noise. We also specify initial scaling matrix with
	anti-correlation between this states.

	>>> k1 = Matern(nu=0.5, length_scale=1.0, length_scale_bounds=(0.01, 100))
	>>> k2 = WhiteKernel(noise_level=1, noise_level_bounds='fixed')
	>>> ms_kernel = MultiStateKernel((k1, k2,),
	...                              np.array([[1,0],[-0.5,1]]),
	...                              [np.array([[-2.0,-2.0],[-2.0,-2.0]]),
	...                              np.array([[2.0,2.0],[2.0,2.0]])])

	See Also
	--------
	sklearn.gaussian_process.kernels.Kernel : base kernel interface
	"""

	def _get_kernel_dict(self):
		return dict([("s" + str(n), kernel) for n, kernel in enumerate(self.state_kernels)])

	def __init__(self, kernels, scale, scale_bounds):
		class ConstantMatrix(Kernel):
			@staticmethod
			def _matrix2flat(matrix):
				return matrix[np.tril_indices_from(matrix)]

			@staticmethod
			def _flat2matrix(flat):
				dimK = flat.shape[0]
				dimX = int(math.floor(math.sqrt(2*dimK)))

				matrix = np.zeros((dimX, dimX))
				matrix[np.tril_indices_from(matrix)] = flat

				return matrix

			def __init__(self, coeffs, coeffs_bounds):
				coeffs = np.asarray(coeffs)
				coeffs[np.triu_indices_from(coeffs, k=1)] = 0
				coeffs_bounds = np.asarray(coeffs_bounds)

				self.params = {'coeffs': coeffs, 'coeffs_bounds': coeffs_bounds}

				if coeffs.ndim == 2:
					matrix = np.tril(coeffs)
					lower, upper = np.tril(coeffs_bounds[0]), np.tril(coeffs_bounds[1])
				else:
					raise ValueError("coeffs has wrong or unsupported dimension")

				self.coeffs = self._matrix2flat(matrix)
				self.coeffs_bounds = np.stack((self._matrix2flat(lower), self._matrix2flat(upper)), 1)

			@property
			def tril(self):
				return self._flat2matrix(self.coeffs)

			@property
			def hyperparameter_coeffs(self):
				return Hyperparameter("coeffs", "numeric", self.coeffs_bounds, self.coeffs.shape[0])

			def get_params(self, deep = True):
				self.params["coeffs"][np.tril_indices_from(self.params["coeffs"])] = self.coeffs
				return self.params

			@property
			def theta(self):
				return self.coeffs

			@property
			def bounds(self):
				return self.coeffs_bounds

			@theta.setter
			def theta(self, theta):
				self.coeffs = theta

			def __call__(self, X, Y=None, eval_gradient=False):
				raise NotImplementedError()
			def diag(self, X):
				raise NotImplementedError()
			def is_stationary(self):
				raise NotImplementedError()

		self.state_kernels = kernels
		self.scale_kernel = ConstantMatrix(scale, scale_bounds)

		kwargs = self._get_kernel_dict()
		kwargs['scale'] = self.scale_kernel

		super(MultiStateKernel, self).__init__(**kwargs)

	def get_params(self, deep=True):
		params = super(MultiStateKernel, self).get_params(deep)
		for name, kernel in self._get_kernel_dict().items():
			del params[name]
		del params['scale']
		if deep:
			del params['scale__coeffs']
			del params['scale__coeffs_bounds']

		params['kernels'] = self.state_kernels

		scale_params = self.scale_kernel.get_params(False)
		params['scale'] = scale_params['coeffs']
		params['scale_bounds'] = scale_params['coeffs_bounds']

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
		n_states = len(self.state_kernels)
		scale = self.scale_kernel.tril

		assert n_featuresX == n_featuresY

		n_features = n_featuresX

		X_reduced, X_reduced_index = np.unique(X[:,1:], return_inverse=True, axis=0)
		if X is Y:
			Y_reduced, Y_reduced_index = X_reduced, X_reduced_index
		else:
			Y_reduced, Y_reduced_index = np.unique(Y[:,1:], return_inverse=True, axis=0)

		n_samples_X_reduced, n_featuresX_reduced = X_reduced.shape
		n_samples_Y_reduced, n_featuresY_reduced = Y_reduced.shape

		states_X = X[:,0].astype(int)
		states_Y = Y[:,0].astype(int)

		related_X = [[]]*n_states
		related_Y = [[]]*n_states
		related_X_reduced = [[]]*n_states
		related_Y_reduced = [[]]*n_states

		for n in range(n_states):
			related_X[n] = (states_X == n)
			related_Y[n] = (states_Y == n)
			related_X_reduced[n] = X_reduced_index[related_X[n]]
			related_Y_reduced[n] = Y_reduced_index[related_Y[n]]

		kernel_pos = [[]]*n_states
		scale_index = np.zeros(shape=scale.shape, dtype=int)
		scale_index[np.tril_indices_from(scale_index)] = np.arange(len(np.tril_indices_from(scale_index)[0]))
		scale_pos = 0
		for k, kernel in enumerate(self.state_kernels):
			begin = scale_pos
			scale_pos += kernel.n_dims
			end = scale_pos
			kernel_pos[k] = (begin, end)

		kernel_value = np.zeros(shape=(n_samples_X_reduced, n_samples_Y_reduced, n_states))
		if eval_gradient:
			kernel_gradient = np.zeros(shape=(n_samples_X_reduced, n_samples_Y_reduced, scale_pos
))

		for k in range(n_states):
			kernel = self.state_kernels[k]
			if eval_gradient:
				kernel_value[...,k], kernel_gradient[...,range(*kernel_pos[k])] = kernel(X_reduced, None, True)
			elif X is Y:
				# See WhiteKernel implementation details
				kernel_value[...,k] = kernel(X_reduced, None, False)
			else:
				kernel_value[...,k] = kernel(X_reduced, Y_reduced, False)

		K = np.zeros(shape=(n_samples_X, n_samples_Y))
		if eval_gradient:
			K_gradient = np.zeros(shape=(n_samples_X, n_samples_Y, self.n_dims))

		for n in range(n_states):
			"""
			Block = Sum_{k=0..min{n,m}} scale_{n, k} scale{m, k} kernel_{k} (X_n, X_m)
			dBlock_{n,m} / dscale_{i,k} = Delta_{n,i} scale_{m,k} kernel_{k} (X_n, X_m)
				+ Delta_{m,i} scale_{n,k} kernel_{k} (X_n, X_m), where k <= min(n,m)
			"""
			for m in range(n_states):
				for k in range(min(n,m) + 1):
					sub_kernel_value = kernel_value[np.ix_(related_X_reduced[n],related_Y_reduced[m])][...,k]
					K[np.ix_(related_X[n],related_Y[m])] += scale[n,k] * scale[m,k] * sub_kernel_value

					if eval_gradient:
						# Fill in gradient with respect to kernel_{k} params
						sub_kernel_gradient = kernel_gradient[np.ix_(related_X_reduced[n], related_Y_reduced[m], range(*kernel_pos[k]))]
						K_gradient[np.ix_(related_X[n],related_Y[m],range(*kernel_pos[k]))] += scale[n,k] * scale[m,k] * sub_kernel_gradient

						nk_index = scale_pos + scale_index[n,k]
						# Fill in gradient with respect to scale matrix params
						K_gradient[np.ix_(related_X[n],related_Y[m],(nk_index,))] += scale[m,k] * sub_kernel_value[...,np.newaxis]
						mk_index = scale_pos + scale_index[m,k]
						K_gradient[np.ix_(related_X[n],related_Y[m],(mk_index,))] += scale[n,k] * sub_kernel_value[...,np.newaxis]

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
		scale = self.scale_kernel.tril

		K_diag = np.zeros(shape=(n_samples_X,))

		states_X = X[:,0].astype(int)

		for n in range(len(self.state_kernels)):
			""" Diag = Sum_{k=0..n} scale_{n, k}^2 kernel_{k} (X_n, X_n) """
			related_X = (states_X == n)

			for k in range(n + 1):
				kernel = self.state_kernels[k]
				K_diag[related_X] += scale[n,k]**2 * kernel.diag(X[related_X,1:])

		return K_diag
