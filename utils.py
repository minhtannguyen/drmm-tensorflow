

__author__ = "rishabgoel"

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
			self.beta = tf.Variable(tf.ones(insize, dtype=tf.float32), name='beta_bn')
		else:
			self.beta = tf.Variable(tf.convert_to_tensor(beta_val_init, dtype = tf.float32), name='beta_bn')

		if mean_init is None:
			self.mean = tf.Variable(tf.ones([insize], dtype=tf.float32), name='mean_bn')
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

		if self.mode == 0:
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
			print('Use Layer-wise BN')
			if self.run_mode == 0:
				now_mean, now_var = tf.nn.moments(input, axes=[0, 2, 3])

				# now_mean_new_shape = self.change_shape(now_mean)
				# now_var = T.sqr(T.mean(T.abs_(input - now_mean_new_shape), axis=(0,2,3)))

				# mean, var update
				self.mean_new = self.momentum * self.mean + (1.0 - self.momentum) * now_mean
				self.var_new = self.momentum * self.var \
						   + (1.0 - self.momentum) * (input_shape[0] / (input_shape[0] - 1) * now_var)

				# self.var_new = self.momentum * self.var + (1.0 - self.momentum) * now_var

				now_mean_4D = tf.case([(tf.not_equal(self.is_train, 0),self.change_shape(now_mean, input_shape))], default =   self.change_shape(self.mean_new, input_shape))
				now_var_4D = tf.case([(tf.not_equal(self.is_train, 0),self.change_shape(now_var, input_shape))], default =  self.change_shape(self.var_new, input_shape))

				now_gamma_4D = self.change_shape(self.gamma, input_shape)
				now_beta_4D = self.change_shape(self.beta, input_shape)

				output = now_gamma_4D * (input - now_mean_4D) / T.sqrt(now_var_4D + self.epsilon) + now_beta_4D

			else:
				now_mean, now_var = tf.nn.moments(input, axes=[0, 2, 3])
				
				# now_mean_new_shape = self.change_shape(now_mean)
				# now_var = T.sqr(T.mean(T.abs_(input - now_mean_new_shape), axis=(0, 2, 3)))

				now_mean_4D = tf.case([(tf.not_equal(self.is_train, 0), self.change_shape(now_mean, input_shape))], default =   self.change_shape(self.mean, input_shape))
				now_var_4D = tf.case([(tf.not_equal(self.is_train, 0), self.change_shape(now_var, input_shape))], default =  self.change_shape(self.var, input_shape))

				now_gamma_4D = self.change_shape(self.gamma, input_shape)
				now_beta_4D = self.change_shape(self.beta, input_shape)

				output = now_gamma_4D * (input - now_mean_4D) / T.sqrt(now_var_4D + self.epsilon) + now_beta_4D

		return output
	# changing shape for CNN mode
	def change_shape(self, vec, input_shape):
		return tf.reshape(tf.tile(vec), multiples  = [input_shape[2] * input_shape[3]]), [input_shape[1], input_shape[2], input_shape[3]])

