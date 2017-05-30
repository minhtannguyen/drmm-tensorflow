

__author__ = "rishabgoel"
"""
Based on theano code by minhtannguyen
"""
import tensorflow as tf
import numpy as np


class BatchNormalization(object):
	def __init__(self, insize, momentum, is_train, mode=0, epsilon=1e-10, gamma_val_init=None, beta_val_init=None,
				 mean_init=None, var_init=None):
		'''
		# params :
		input_shape :
			when mode is 0, we assume 2D input. (mini_batch_size, # features)
			when mode is 1, we assume 4D input. (mini_batch_size, # of channel, # row, # column)
		mode :
			0 : feature-wise mode (normal BN)
			1 : window-wise mode (CNN mode BN)
		momentum : momentum for exponential average.
		'''
		self.insize = insize
		self.mode = mode
		self.momentum = momentum
		self.is_train = is_train
		self.run_mode = 0  # run_mode :
		self.epsilon = epsilon

		# random setting of gamma and beta, setting initial mean and std
		# rng = np.random.RandomState(int(time.time()))
		# self.gamma = theano.shared(np.asarray(
		#     rng.uniform(low=-1.0 / math.sqrt(self.insize), high=1.0 / math.sqrt(self.insize), size=(insize)),
		#     dtype=theano.config.floatX), name='gamma', borrow=True)
		if gamma_val_init is None:
			self.gamma = tf.Variable(tf.ones(insize, dtype=tf.float32), name='gamma_bn')
		else:
			self.gamma = tf.Variable(tf.convert_to_tensor(gamma_val_init, dtype = tf.float32), name='gamma_bn')

		if beta_val_init is None:
			self.beta = tf.Variable(tf.zeros(insize, dtype=tf.float32), name='beta_bn')
		else:
			self.beta = tf.Variable(tf.convert_to_tensor(beta_val_init, dtype = tf.float32), name='beta_bn')

		if mean_init is None:
			self.mean = tf.Variable(tf.zeros([insize], dtype=tf.float32), name='mean_bn')
		else:
			self.mean = tf.Variable(tf.convert_to_tensor(gamma_val_init, dtype = tf.float32), name='mean_bn')

		if var_init is None:
			self.var = tf.Variable(tf.ones([insize], dtype=tf.float32), name='var_bn')
		else:
			self.var = tf.Variable(tf.convert_to_tensor(gamma_val_init, dtype = tf.float32), name='var_bn')

		# parameter save for update
		self.params = [self.gamma, self.beta]

	def set_runmode(self, run_mode):
		self.run_mode = run_mode

	def get_result(self, input, input_shape):
		# returns BN result for given input.
		# self.mode = 1
		if self.mode == 0:
			# self.run_mode = 1
			print('Use Feature-wise BN')
			if self.run_mode == 0:
				now_mean, now_var = tf.nn.moments(input, axes=[0])
				# now_var = tf.square(input, axis=0)
				now_normalize = (input - now_mean) / tf.sqrt(now_var + self.epsilon)  # should be broadcastable..
				output = self.gamma * now_normalize + self.beta
				# mean, var update
				self.mean = self.momentum * self.mean + (1.0 - self.momentum) * now_mean
				self.var = self.momentum * self.var \
						   + (1.0 - self.momentum) * (input_shape[0] / (input_shape[0] - 1) * now_var)
			elif self.run_mode == 1:
				now_mean, now_var = tf.nn.moments(input, axes=[0])
				# now_mean = T.mean(input, axis=0)
				# now_var = T.var(input, axis=0)
				output = self.gamma * (input - now_mean) / tf.sqrt(now_var + self.epsilon) + self.beta
			else:
				now_mean = self.mean
				now_var = self.var
				output = self.gamma * (input - now_mean) / tf.sqrt(now_var + self.epsilon) + self.beta

		else:
			# in CNN mode, gamma and beta exists for every single channel separately.
			# for each channel, calculate mean and std for (mini_batch_size * row * column) elements.
			# then, each channel has own scalar gamma/beta parameters.
			print('Use Layer-wise BN,t')
			# self.run_mode = 0
			if self.run_mode == 0:
				print "tensorflow"
				now_mean, now_var = tf.nn.moments(input, axes=[0, 2, 3])
				self.now_mean, self.now_var = now_mean, now_var
				# now_mean_new_shape = self.change_shape(now_mean)
				# now_var = T.sqr(T.mean(T.abs_(input - now_mean_new_shape), axis=(0,2,3)))

				# mean, var update
				self.mean_new = self.momentum * self.mean + (1.0 - self.momentum) * now_mean
				self.var_new = self.momentum * self.var \
						   + (1.0 - self.momentum) * (input_shape[0] / (input_shape[0] - 1) * now_var)

				# self.var_new = self.momentum * self.var + (1.0 - self.momentum) * now_var
				# print self.is_train
				self.is_train = 1
				now_mean_4D = tf.cond(tf.not_equal(self.is_train, 0), lambda: self.change_shape(now_mean, input_shape), lambda: self.change_shape(self.mean_new, input_shape))
				now_var_4D = tf.cond(tf.not_equal(self.is_train, 0), lambda: self.change_shape(now_var, input_shape), lambda: self.change_shape(self.var_new, input_shape))

				self.now_mean_4D =now_mean_4D
				self.now_var_4D = now_var_4D
				now_gamma_4D = self.change_shape(self.gamma, input_shape)
				now_beta_4D = self.change_shape(self.beta, input_shape)

				output = now_gamma_4D * (input - now_mean_4D) / tf.sqrt(now_var_4D + self.epsilon) + now_beta_4D

			else:
				print "tensorflow1"
				now_mean, now_var = tf.nn.moments(input, axes=[0, 2, 3])
				# print self.is_train
				# now_mean_new_shape = self.change_shape(now_mean)
				# now_var = T.sqr(T.mean(T.abs_(input - now_mean_new_shape), axis=(0, 2, 3)))
				# self.is_train = 1
				if self.is_train:
					now_mean_4D = self.change_shape(now_mean, input_shape)
					now_var_4D = self.change_shape(now_var, input_shape)
				else:
					now_mean_4D = self.change_shape(self.mean, input_shape)
					now_var_4D = self.change_shape(self.var, input_shape)

				now_gamma_4D = self.change_shape(self.gamma, input_shape)
				now_beta_4D = self.change_shape(self.beta, input_shape)

				output = now_gamma_4D * (input - now_mean_4D) / tf.sqrt(now_var_4D + self.epsilon) + now_beta_4D

		return output
	# changing shape for CNN mode
	def change_shape(self, vec, input_shape):
		return tf.reshape(tf.reshape(tf.tile(tf.reshape(vec, [-1,1]), [1, input_shape[2] * input_shape[3]]), [-1]),
			[input_shape[1], input_shape[2], input_shape[3]])

class HiddenLayerInSoftmax(object):
	def __init__(self, input_lab, input_unl, n_in, n_out, W_init=None, b_init=None, input_clean=None):
		"""
		Typical hidden layer of a MLP: units are fully-connected and have
		sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
		and the bias vector b is of shape (n_out,).

		NOTE : The nonlinearity used here is tanh

		Hidden unit activation is given by: tanh(dot(input,W) + b)

		:type input_lab, input_unl, input_clean: theano.tensor.dmatrix
		:param input_lab, input_unl, input_clean: a symbolic tensor of shape (n_examples, n_in)

		:type n_in: int
		:param n_in: dimensionality of input

		:type n_out: int
		:param n_out: number of hidden units

		"""
		if W_init is None:
			W_bound = tf.sqrt(tf.cast(1/n_in, tf.float32))
			# W_val = np.asarray(np.random.uniform(low=-W_bound, high=W_bound, size=(n_in, n_out)),dtype=theano.config.floatX)
			self.W = tf.Variable(tf.random_uniform([n_in, n_out], minval = - W_bound, maxval = W_bound, dtype = tf.float32), name='W')

			# self.W = theano.shared(value=numpy.zeros((n_in, n_out),
			#                                          dtype=theano.config.floatX) + 1e-8,
			#                        name='W', borrow=True)
		else:
			self.W = tf.Variable(tf.conver_to_tensor(W_init), name='W')
		# initialize the baises b as a vector of n_out 0s
		if b_init is None:
			self.b = tf.Variable(tf.zeros([n_out,], dtype=tf.float32) + 1e-8, name='b')
		else:
			self.b = tf.Variable(tf.convert_to_tensor(b_init), name='b')

		self.output_lab = None
		self.output = None
		self.output_clean = None

		if input_lab is not None:
			self.output_lab = tf.add(tf.matmul(input_lab, self.W),self.b)

		if input_unl is not None:
			self.output = tf.add(tf.matmul(input_unl, self.W),self.b)

		if input_clean is not None:
			self.output_clean = tf.add(tf.matmul(input_clean, self.W),self.b)

		# parameters of the model
		self.params = [self.W, self.b]




class SoftmaxNonlinearity(object):
	def __init__(self, input_lab, input_unl, input_clean=None):
		self.gammas = None
		self.gammas_clean = None
		self.gammas_lab = None
		self.y_pred = None
		self.y_pred_clean = None
		self.y_pred_lab = None

		if input_lab != None:
			self.gammas_lab = tf.nn.softmax(input_lab)
			self.y_pred_lab = tf.argmax(self.gammas_lab, axis=1)

		if input_unl != None:
			self.gammas = tf.nn.softmax(input_unl)
			self.y_pred = tf.argmax(self.gammas, axis=1)

		if input_clean != None:
			# compute vector of class-membership probabilities in symbolic form
			self.gammas_clean =  tf.nn.softmax(input_clean)
			# compute prediction as class whose probability is maximal in
			# symbolic form
			self.y_pred_clean = tf.argmax(self.gammas_clean, axis=1)

	def negative_log_likelihood(self, y):
		"""Return the mean of the negative log-likelihood of the prediction
		of this model under a given target distribution.

		.. math::

			\frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
			\frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
				\ell (\theta=\{W,b\}, \mathcal{D})

		:type y: theano.tensor.TensorType
		:param y: corresponds to a vector that gives for each example the
				  correct label

		Note: we use the mean instead of the sum so that
			  the learning rate is less dependent on the batch size
		"""
		# y.shape[0] is (symbolically) the number of rows in y, i.e.,
		# number of examples (call it n) in the minibatch
		# T.arange(y.shape[0]) is a symbolic vector which will contain
		# [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
		# Log-Probabilities (call it LP) with one row per example and
		# one column per class LP[T.arange(y.shape[0]),y] is a vector
		# v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
		# LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
		# the mean (across minibatch examples) of the elements in v,
		# i.e., the mean log-likelihood across the minibatch.
		return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = self.gammas_lab))
		# return -T.mean(T.log(self.gammas_lab)[T.arange(y.shape[0]), y])

	def errors(self, y):
		"""Return a float representing the number of errors in the minibatch
		over the total number of examples of the minibatch ; zero one
		loss over the size of the minibatch

		:type y: theano.tensor.TensorType
		:param y: corresponds to a vector that gives for each example the
				  correct label
		"""

		# check if y has same dimension of y_pred
		# print self.y_pred.get_shape(), y.get_shape
		# self.y
		if y.get_shape() != self.y_pred_lab.get_shape():
			raise TypeError(
				'y should have the same shape as self.y_pred',
				('y', y.type, 'y_pred', self.y_pred.type)
			)
		# check if y is of the correct datatype
		if y.dtype==tf.int64 and self.y_pred_lab.dtype == tf.int64:
			# the T.neq operator returns a vector of 0s and 1s, where 1
			# represents a mistake in prediction
			# self.y_pred.dtype = tf.cast(self.)
			# d = tf.not_equal(self.y_pred_lab, y)
			return tf.reduce_mean(tf.cast(tf.not_equal(self.y_pred_lab, y), dtype=tf.float32))
		else:
			raise NotImplementedError()
