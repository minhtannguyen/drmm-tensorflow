"""
This file implements the a single layer of a model
"""

__author__ = 'rishabgoel'
"""
Based on theano code by minhtannguyen
"""


import matplotlib as mpl

mpl.use('Agg')

import numpy as np

np.set_printoptions(threshold='nan')


from utils import BatchNormalization

import tensorflow as tf


class Layer():

	"""
	Arguments : 
		internal variables:
		`data_4D_lab`:  labeled images
		`data_4D_unl`:  unlabeled images
		`data_4D_unl_clean`: unlabeled images used for the clean pass without noise. This is used when we add noise to the
								the model to make it robust. We are not doing this now.
		`noise_weight`:control how much noise we want to add to the model. Since we don't add noise to the model, we always 
					   noise_weight to 0
		`noise_std`: the standard deviation of the noise
		`K`:           Number of the rendering matrices lambdas_t
		`M`:           Latent dimensionality (set to 1 when the DRMM is applied at the patch level. In general, can be set 
					   to any value, and we save this for the future work)
		`W`:           Width of the input
		`H`:           Height of the input
		`w`:           Width of the rendering matrices
		`h`:           Height of the rendering matrices
		`Cin`:         Number of channels of the input = Number of channels of the rendering matrices
		`Ni`:          Number of input images
		`momentum_bn`: control how to update the batch mean and var in batch normalization (BatchNorm) which will be used in testing
		`is_train`:     {0,1} where 0: testing mode and 1: training mode
		`lambdas_t_val_init`: Initial values for rendering matrices
		`gamma_val_init`: Initial values for the correction term gamma in BatchNorm
		`beta_val_init`: Initial values for the correction term beta in BatchNorm
		`mean_bn_init`: Initial values for the batch mean in BatchNorm
		`var_bn_init`: Initial values for the batch variance in BatchNorm
		`prun_mat_init`: Initial values for the neuron pruning masking matrices
		`prun_synap_mat_init`: Initial values for the synaptic pruning masking matrices
		`sigma_dn_init`: Initial values for the sigma term in DivNorm
		`b_dn_init`: Initial values for the bias term in DivNorm
		`alpha_dn_init`: Initial values for the alpha term in DivNorm
		`pool_t_mode`: Pooling mode={'max_t','mean_t`,None}
		`border_mode`: {'valid`, 'full', `half`} mode for the convolutions
		`nonlin`: {`relu`, `abs`, None}
		`mean_pool_size`: size of the mean pooling layer before the softmax regression
		`max_condition_number`: a condition number to make computations more stable. Set to 1.e3
		`xavier_init`: {True, False} use Xavier initialization or not
		`is_noisy`: {True, False} add noise to the model or not. Since we don't add noise to the model, we always set this 
					to False
		`is_bn_BU`: {True, False} do batch normalization in the bottom-up E step or not
		`epsilon`: used to avoid divide by 0
		`momentum_pi_t`: used to update pi_t during training (Exponential Moving Average (EMA) update)
		`momentum_pi_a`: used to update pi_a during training (EMA update)
		`momentum_pi_ta`: used to update pi_ta during training (EMA update)
		`momentum_pi_synap`: used to update pi_synap during training (EMA update)
		`is_prun`: {True, False} apply neuron pruning or not
		`is_prun_synap`: {True, False} apply synaptic pruning or not
		`is_dn`: {True, False} apply divisive normalization (DivNorm) or not
		`update_mean_var_with_sup`: {True, False} if True, only use data_4D_lab to update the mean and var in BatchNorm
	"""
	def __init__(self, data_4D_lab,
				 data_4D_unl,
				 data_4D_unl_clean,
				 noise_weight,
				 noise_std,
				 K, M, W, H, w, h, Cin, Ni,
				 momentum_bn, is_train,
				 lambdas_t_val_init=None, 
				 gamma_val_init=None, beta_val_init=None,
				 prun_mat_init=None, prun_synap_mat_init=None,
				 mean_bn_init=None, var_bn_init=None,
				 pool_t_mode='max_t', border_mode='VALID', nonlin='relu', 
				 mean_pool_size=[2, 2],
				 max_condition_number=1.e3,
				 weight_init="xavier",
				 is_noisy=False,
				 # is_noisy=True,
				 is_bn_BU=False,
				 # is_bn_BU=True,
				 epsilon=1e-10,
				 momentum_pi_t=0.99,
				 momentum_pi_a=0.99,
				 momentum_pi_ta=0.99,
				 momentum_pi_synap=0.99,
				 # is_prun=False,
				 is_prun=True,
				 # is_prun_synap=False,
				 is_prun_synap=True,
				 # is_dn=False,
				 is_dn=True,
				 sigma_dn_init=None,
				 b_dn_init=None,
				 alpha_dn_init=None,
				 # update_mean_var_with_sup=False,
				 update_mean_var_with_sup=True, 
				 name = "conv_layer"):
		self.name = name
		self.K = K  # number of lambdas_t/filter
		self.M = M  # latent dimensionality. Set to 1 for our model
		self.data_4D_lab = data_4D_lab  # labeled data
		self.data_4D_unl = data_4D_unl  # unlabeled data
		self.data_4D_unl_clean = data_4D_unl_clean  # unlabeled data used for the clean path
		self.noise_weight = noise_weight  # control how much noise we want to add to the model. Set to 0 for our model
		self.noise_std = noise_std  # the standard deviation of the noise
		self.Ni = Ni  # no. of labeled examples = no. of unlabeled examples
		self.w = w  # width of filter
		self.h = h  # height of filter
		self.Cin = Cin  # number of channels in the image
		self.D = self.h * self.w * self.Cin  # patch size
		self.W = W  # width of image
		self.H = H  # height of image
		if border_mode == 'VALID':  # convolution mode. Output size is smaller than input size
			self.Np = (self.H - self.h + 1) * (self.W - self.w + 1)  # no. of patches per image
			self.latents_shape = (self.Ni, self.K, self.H - self.h + 1, self.W - self.w + 1)
		elif border_mode == 'HALF':  # convolution mode. Output size is the same as input size
			self.Np = self.H * self.W  # no. of patches per image
			self.latents_shape = (self.Ni, self.K, self.H, self.W)
		elif border_mode == 'FULL':  # # convolution mode. Output size is greater than input size
			self.Np = (self.H + self.h - 1) * (self.W + self.w - 1)  # no. of patches per image
			self.latents_shape = (self.Ni, self.K, self.H + self.h - 1, self.W + self.w - 1)
		else:
			raise
		print "latent_shape : ", self.latents_shape, data_4D_lab.get_shape()
		self.N = self.Ni * self.Np  # total no. of patches and total no. of hidden units
		self.mean_pool_size = mean_pool_size  # size of the mean pooling layer before the softmax regression

		self.lambdas_t_val_init = lambdas_t_val_init  # Initial values for rendering matrices
		self.gamma_val_init = gamma_val_init  # Initial values for the correction term gamma in BatchNorm
		self.beta_val_init = beta_val_init  # Initial values for the correction term beta in BatchNorm
		self.prun_synap_mat_init = prun_synap_mat_init  # Initial values for the synaptic pruning masking matrices
		self.prun_mat_init = prun_mat_init  # Initial values for the neuron pruning masking matrices
		self.sigma_dn_init = sigma_dn_init  # Initial values for the sigma term in DivNorm
		self.mean_bn_init = mean_bn_init  # Initial values for the batch mean in BatchNorm
		self.var_bn_init = var_bn_init  # Initial values for the batch variance in BatchNorm
		self.b_dn_init = b_dn_init  # Initial values for the bias term in DivNorm
		self.alpha_dn_init = alpha_dn_init  # Initial values for the alpha term in DivNorm

		self.pool_t_mode = pool_t_mode  # Pooling mode={'max_t','mean_t`,None}
		self.nonlin = nonlin  # {`relu`, `abs`, None}
		
		self.border_mode = border_mode  # {'valid`, 'full', `half`} mode for the convolutions
		self.max_condition_number = max_condition_number  # a condition number to make computations more stable. Set to 1.e3
		self.weight_init = weight_init  # type of weight initialisation
		self.is_noisy = is_noisy  # {True, False} add noise to the model or not.
		self.is_bn_BU = is_bn_BU  # {True, False} do batch normalization in the bottom-up E step or not
		self.momentum_bn = momentum_bn  # control how to update the batch mean and var in batch normalization (BatchNorm)
		self.momentum_pi_t = momentum_pi_t  # used to update pi_t during training (Exponential Moving Average (EMA) update)
		self.momentum_pi_a = momentum_pi_a  # used to update pi_a during training (EMA update)
		self.momentum_pi_ta = momentum_pi_ta  # used to update pi_ta during training (EMA update)
		self.momentum_pi_synap = momentum_pi_synap  # used to update pi_synap during training (EMA update)
		self.is_train = is_train  # {0,1} where 0: testing mode and 1: training mode
		self.epsilon = epsilon  # used to avoid divide by 0
		self.is_prun = is_prun  # {True, False} apply neuron pruning or not
		self.is_prun_synap = is_prun_synap  # {True, False} apply synaptic pruning or not
		self.is_dn = is_dn  # {True, False} apply divisive normalization (DivNorm) or not
		self.update_mean_var_with_sup = update_mean_var_with_sup  # {True, False} if True, only use data_4D_lab to update
																	# the mean and var in BatchNorm
		self._initialize()

	def _initialize(self):
		#
		# initialize the model parameters.
		# all parameters involved in the training are collected in self.params for the gradient descent step

		# add noise to the input if is_noisy
		if self.is_noisy:
			if self.data_4D_unl is not None:
				self.data_4D_unl = self.data_4D_unl + \
								   self.noise_weight * tf.random_normal([self.Ni, self.Cin, self.H, self.W],
																		mean=0.0, stddev=self.noise_std)
			if self.data_4D_lab is not None:
				self.data_4D_lab = self.data_4D_lab + \
								   self.noise_weight * tf.random_normal([self.Ni, self.Cin, self.H, self.W],
																		mean=0.0, stddev=self.noise_std)


		# initialize t and a priors
		print self.latents_shape[1:]
		self.pi_t = tf.Variable(tf.ones(self.latents_shape[1:], dtype = tf.float32), name = "pi_t")
		self.pi_a = tf.Variable(tf.ones(self.latents_shape[1:], dtype = tf.float32), name = "pi_a")
		self.pi_a_old = tf.Variable(tf.ones(self.latents_shape[1:], dtype = tf.float32), name = "pi_a_old")
		self.pi_ta = tf.Variable(tf.ones(self.latents_shape[1:], dtype = tf.float32), name = "pi_ta")
		

		# initialize the pruning masking matrices
		if self.prun_mat_init is None:
			self.prun_mat = tf.Variable(tf.ones(self.latents_shape[1:], dtype = tf.float32), name = "prun_mat")
		else:
			self.prun_mat = tf.Variable(tf.convert_to_tensor(np.asarray(self.prun_mat_init), dtype = tf.float32, name = "prun_mat"))
			
		if self.prun_synap_mat_init is None:
			self.prun_synap_mat = tf.Variable(tf.ones([self.K, self.Cin, self.h, self.w]), dtype=tf.float32,name='prun_synap_mat')
		else:
			self.prun_synap_mat = tf.Variable(tf.convert_to_tensor(np.asarray(self.prun_synap_mat_init)), dtype=tf.float32, name='prun_mat')

		# initialize synapse prior (its probability to be ON or OFF)
		if self.is_prun_synap:
			self.pi_synap = tf.Variable(tf.ones([self.K, self.Cin, self.h, self.w], dtype = tf.float32), name = 'pi_synap')
			self.pi_synap_old = tf.Variable(tf.ones([self.K, self.Cin, self.h, self.w], dtype = tf.float32), name = 'pi_synap_old')
			
		# pi_t_final and pi_a_final are used after training for sampling
		self.pi_t_final = tf.Variable(tf.ones(self.latents_shape[1:], dtype = tf.float32), name = "pi_t_final")
		self.pi_a_final = tf.Variable(tf.ones(self.latents_shape[1:], dtype = tf.float32), name = "pi_a_final")
		
		# initialize the lambdas_t
		# if initial values for lambdas_t are not provided, randomly initialize lambdas_t
		initialised = 0
		if self.lambdas_t_val_init is None:
			if self.weight_init == "xavier":
				initialised = 1
				self.lambdas_t = tf.get_variable(shape = [self.K, self.D, self.M],name = "lambdas_t"+self.name,  initializer = tf.contrib.layers.xavier_initializer())
			else:
				lambdas_t_value = np.random.randn(self.K, self.D, self.M) / \
								np.sqrt(self.max_condition_number)
		else:
			lambdas_t_value = self.lambdas_t_val_init
		if initialised == 0:
			self.lambdas_t = tf.Variable(tf.convert_to_tensor(lambdas_t_value, dtype=tf.float32), name='lambdas_t')

		# Initialize BatchNorm
		if self.is_bn_BU:
			self.bn_BU = BatchNormalization(insize=self.K, mode=1, momentum=self.momentum_bn, is_train=self.is_train,
											epsilon=self.epsilon,
											gamma_val_init=self.gamma_val_init, beta_val_init=self.beta_val_init,
											mean_init=self.mean_bn_init, var_init=self.var_bn_init)

			self.params = [self.lambdas_t, self.bn_BU.gamma, self.bn_BU.beta]
		else:
			self.params = [self.lambdas_t, ]

		# Initialize the output
		self.output_lab = None
		self.output_clean = None
		self.output_1 = None
		self.output_2 = None

		# Initialize the sigma parameter in DivNorm
		if self.sigma_dn_init is None:
			self.sigma_dn = tf.Variable(0.5*tf.ones([1], dtype=tf.float32), name='sigma_dn')
		else:
			self.sigma_dn = tf.Variable(tf.convert_to_tensor(self.sigma_dn_init, dtype=tf.float32), name='sigma_dn')

		# Initialize the bias in DivNorm
		if self.b_dn_init is None:
			self.b_dn = tf.Variable(0. * tf.ones([1], dtype=tf.float32), name='b_dn')
		else:
			self.b_dn = tf.Variable(tf.convert_to_tensor(self.b_dn_init, dtype=tf.float32), name='b_dn')

		# Initialize the alpha parameter in DivNorm
		if self.alpha_dn_init is None:
			self.alpha_dn = tf.Variable(tf.ones([1], dtype=tf.float32), name='alpha_dn')
		else:
			self.alpha_dn = tf.Variable(tf.convert_to_tensor(self.alpha_dn_init, dtype=tf.float32), name='alpha_dn')

		if self.is_dn:
			self.params.append(self.sigma_dn)
			self.params.append(self.alpha_dn)
			self.params.append(self.b_dn)

	def get_important_latents_BU(self, input, betas):
		""""This function is used in the _E_step_Bottom_Up to compute latent representations
		
		Return:
			latents_before_BN: activations after convolutions
			latents: activations after BatchNorm/DivNorm
			max_over_a_mask: masking tensor results from ReLU
			max_over_t_mask: masking tensor results from max-pooling
			latents_masked: activations after BatchNorm/DivNorm masked by a and t
			masked_mat: max_over_t_mask * max_over_a_mask
			output: output of the layer, a.k.a. the downsampled activations
			mask_input: the input masked by the ReLU
			scale_s: the scale latent variable in DivNorm
			latents_demeaned: activations after convolutions whose means are removed
			latents_demeaned_squared: (activations after convolutions whose means are removed)^2

		"""
		# compute the activations after convolutions
		print "i am called"
		# print "here",self.border_mode,input.get_shape(), betas.get_shape()
		# self.betas111 = betas
		# self.iiii = input
		if self.border_mode != 'FULL':
			if self.border_mode == "HALF":
				padded_imgs,_ = self.pad_images(input, input.get_shape(), betas.get_shape().as_list(),"HALF")
				latents_before_BN = tf.nn.conv2d(tf.transpose(padded_imgs, [0,2,3,1]), tf.transpose(betas,[2,3,1,0]), strides=[1, 1, 1, 1], padding="VALID")
			else:
				latents_before_BN = tf.nn.conv2d(tf.transpose(input, [0,2,3,1]), tf.transpose(betas,[2,3,1,0]), strides=[1, 1, 1, 1], padding="VALID")
			latents_before_BN = tf.transpose(latents_before_BN, [0, 3, 1, 2])
		else:
			latents_before_BN = tf.nn.conv2d_transpose(tf.transpose(input, [0,2,3,1]), tf.transpose(betas,[2,3,0,1])[::-1,::-1,:,:], strides=[1, 1, 1, 1], padding="VALID",
													   output_shape=[self.latents_shape[0], self.latents_shape[2],
																	 self.latents_shape[3], self.latents_shape[1]])
			latents_before_BN = tf.transpose(latents_before_BN, [0, 3, 1, 2])
		# self.latents_before_BN = latents_before_BN
		# print latents_before_BN.get_shape()
		# do batch normalization or divisive normalization
		if self.is_bn_BU: # do batch normalization
			print "entered in here"
			latents_after_BN = self.bn_BU.get_result(input=latents_before_BN, input_shape=self.latents_shape)
			self.latents_after_BN_lab = latents_after_BN
			scale_s = tf.ones(latents_before_BN.get_shape().as_list(), dtype = tf.float32)
			latents_demeaned = latents_before_BN
			latents_demeaned_squared = latents_demeaned ** 2
		elif self.is_dn: # do divisive normalization
			print "entered in here dn"
			filter_for_norm_local = tf.Variable(tf.ones((1, self.K, self.h, self.w), dtype=tf.float32), name='filter_norm_local')
			# if self.border_mode != 'FULL':
			latents_before_BN_padded,_ = self.pad_images(latents_before_BN, latents_before_BN.get_shape(), [1, self.K, self.h, self.w],"HALF")
			print latents_before_BN_padded.get_shape(),"go"
			sum_local = tf.nn.conv2d(tf.transpose(latents_before_BN_padded, [0,2,3,1]), tf.transpose(filter_for_norm_local,[2,3,1,0]), strides=[1, 1, 1, 1], padding="VALID")
			sum_local = tf.transpose(sum_local, [0, 3, 1, 2])

			# self.latents_before_BN = sum_local
			# else:
			# 	latents_before_BN = tf.nn.conv2d_transpose(tf.transpose(input, [0,2,3,1]), tf.transpose(betas,[2,3,0,1])[::-1,::-1,:,:], strides=[1, 1, 1, 1], padding="VALID",
			# 											   output_shape=[self.latents_shape[0], self.latents_shape[2],
			# 															 self.latents_shape[3], self.latents_shape[1]])
			# 	latents_before_BN = tf.transpose(latents_before_BN, [0, 3, 1, 2])
			# sum_local = tf.nn.conv2d(
			# 	data_format = "NCHW",
			# 	strides = [1,1,1,1],
			# 	input=latents_before_BN,
			# 	filter=filter_for_norm_local,
			# 	padding='half'
			# )

			mean_local = sum_local/(self.K*self.h*self.w)
			# print tf.tile(mean_local, [1,self.K,1,1]).get_shape(),"uo"

			latents_demeaned = latents_before_BN - tf.tile(mean_local, [1,self.K,1,1])

			latents_demeaned_squared = latents_demeaned**2

			latents_demeaned_squared_padded,_ = self.pad_images(latents_demeaned_squared, latents_demeaned_squared.get_shape(), [1, self.K, self.h, self.w],"HALF")
			print latents_demeaned_squared_padded.get_shape(),"uo"
			norm_local = tf.nn.conv2d(tf.transpose(latents_demeaned_squared_padded, [0,2,3,1]), tf.transpose(filter_for_norm_local,[2,3,1,0]), strides=[1, 1, 1, 1], padding="VALID")
			norm_local = tf.transpose(norm_local, [0, 3, 1, 2])

			# norm_local = tf.nn.conv2d(
			# 	data_format = "NCHW",
			# 	strides = [1,1,1,1],
			# 	input=latents_demeaned_squared,
			# 	filter=filter_for_norm_local,
			# 	padding='half'
			# )
			norm_local = norm_local/(self.K * self.h * self.w)
			norm_local = tf.tile(norm_local, [1,self.K,1,1])

			scale_s = (tf.reshape(tf.tile((self.alpha_dn + 1e-10), [self.K]),[1,-1,1,1])
					   + norm_local/((tf.reshape(tf.tile((self.sigma_dn + 1e-5), [self.K]),[1, -1, 1, 1])) ** 2)) / 2.

			# print scale_s.get_shape()
			latents_after_BN = (latents_demeaned / tf.sqrt(scale_s)) + tf.reshape(tf.tile(self.b_dn, [self.K]),[1,-1,1,1])

		else:
			latents_after_BN = latents_before_BN
			scale_s = tf.ones(latents_before_BN.get_shape())
			latents_demeaned = latents_before_BN
			latents_demeaned_squared = latents_demeaned ** 2

		print latents_after_BN.get_shape(), self.prun_mat.get_shape()
		latents = latents_after_BN * self.prun_mat # masking the activations by the neuron pruning mask.
													# self.prun_mat is all 1's if no neuron pruning

		mask_input = tf.cast(tf.greater(input, 0.), tf.float32) # find positive elements in the input

		# find activations survive after max over a
		if self.nonlin == 'relu':
			max_over_a_mask = tf.cast(tf.greater(latents, 0.), tf.float32)
		else:
			max_over_a_mask = tf.cast(tf.ones(latents.get_shape()), tf.float32)
		print "max_over_mask",max_over_a_mask.get_shape()
		# find activations survive after max over t
		if self.pool_t_mode == 'max_t' and self.nonlin == 'relu':
			print('Do max over t')
			# self.to_see = tf.transpose(tf.image.resize_nearest_neighbor(tf.nn.max_pool(tf.transpose(latents, [0,2,3,1]), [1, 2, 2, 1], strides=[1,2,2,1], padding='VALID'),
															# [self.latents_shape[2], self.latents_shape[3]]),[0,3,1,2])
			# # self.to_see = tf.greater_equal(latents,
			# 								   tf.transpose(tf.image.resize_nearest_neighbor(tf.nn.max_pool(tf.transpose(latents, [0,2,3,1]), [1, 2, 2, 1], strides=[1,2,2,1], padding='VALID'),
			# 												[self.latents_shape[2], self.latents_shape[3]]),
			# 												[0,3,1,2]))
			self.to_see = latents
			# max_over_t_mask = tf.gradients(tf.reduce_sum(tf.nn.max_pool(tf.transpose(latents, [0,2,3,1]), [1, 2, 2, 1], strides=[1,2,2,1], padding='VALID')), latents)[0]
			# max_over_t_mask = tf.equal(latents,
			# 								   tf.transpose(tf.image.resize_nearest_neighbor(tf.nn.max_pool(tf.transpose(latents, [0,2,3,1]), [1, 2, 2, 1], strides=[1,2,2,1], padding='VALID'),
			# 												[self.latents_shape[2], self.latents_shape[3]]),
			# 												[0,3,1,2]))
			max_over_t_mask = tf.equal(latents,
											   tf.transpose(tf.image.resize_nearest_neighbor(tf.nn.max_pool(tf.transpose(latents, [0,2,3,1]), [1, 2, 2, 1], strides=[1,2,2,1], padding='VALID'),
															[self.latents_shape[2], self.latents_shape[3]]),
															[0,3,1,2]))
			max_over_t_mask = tf.cast(max_over_t_mask, dtype=tf.float32)
			self.ttt = max_over_t_mask
		elif self.pool_t_mode == 'max_t' and self.nonlin == 'abs': # still in the beta state
			# print('Do max over t')
			latents_abs = tf.abs(latents)
			self.latents_abs = latents_abs
			# max_over_t_mask = tf.gradients(tf.reduce_sum(tf.nn.max_pool(tf.transpose(latents_abs, [0,2,3,1]), [1, 2, 2, 1], strides=[1,2,2,1], padding='VALID')), latents_abs)[0]
			# self.latents_abs = max_over_t_mask
			# max_over_t_mask = tf.equal(latents_abs,
			# 								   tf.transpose(tf.image.resize_nearest_neighbor(tf.nn.max_pool(tf.transpose(latents_abs, [0, 2, 3, 1]), [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'),
			# 									   [self.latents_shape[2], self.latents_shape[3]]),
			# 												[0, 3, 1, 2]))
			# max_over_t_mask = tf.cast(latents_abs, dtype=tf.float32)
			max_over_t_mask = tf.equal(latents_abs,
											   tf.transpose(tf.image.resize_nearest_neighbor(tf.nn.max_pool(tf.transpose(latents_abs, [0, 2, 3, 1]), [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'),
												   [self.latents_shape[2], self.latents_shape[3]]),
															[0, 3, 1, 2]))
			max_over_t_mask = tf.cast(max_over_t_mask, dtype=tf.float32)
			self.ttt = max_over_t_mask
		else:
			# print('No max over t')
			# compute latents masked by a
			max_over_t_mask = tf.cast(tf.ones_like(latents), dtype=tf.float32)

		# mask the activations by t and a
		if self.nonlin == 'relu':
			latents_masked = tf.nn.relu(latents) * max_over_t_mask  # * max_over_a_mask
		elif self.nonlin == 'abs':
			latents_masked = tf.abs(latents) * max_over_t_mask
		else:
			latents_masked = latents * max_over_t_mask

		# find activations survive after max over a and t
		masked_mat = max_over_t_mask * max_over_a_mask  # * max_over_a_mask


		# downsample the activations
		if self.pool_t_mode == 'max_t':
			output = tf.nn.avg_pool(tf.transpose(latents_masked, [0,2,3,1]), [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
			output = output * 4.0
			output = tf.transpose(output, [0,3,1,2])
		elif self.pool_t_mode == 'mean_t':
			output = tf.nn.avg_pool(tf.transpose(latents_masked, [0,2,3,1]), [1] + self.mean_pool_size+[1], strides=[1]+self.mean_pool_size+[1], padding='VALID')
			output = tf.transpose(output, [0, 3, 1, 2])
		else:
			output = latents_masked


		return latents_before_BN, latents, max_over_a_mask, max_over_t_mask, latents_masked, masked_mat, output, mask_input, scale_s, latents_demeaned, latents_demeaned_squared


	def EBottomUp(self):
		"""
		E-step bottom-up infers the latents in the images
		"""

		# reshape lambdas_t into the filter

		self.betas = tf.transpose(self.lambdas_t, [0, 2, 1])
		betas = tf.reshape(tf.squeeze(self.betas,[1]), [self.K, self.Cin, self.h, self.w])
		# betas = tf.reshape(self.betas[:, 0, :], shape=(self.h, self.w, self.Cin, self.K))
		print betas.get_shape(),"here", self.prun_synap_mat

		betas = betas * self.prun_synap_mat

		# run bottom up for labeled examples.
		if self.is_bn_BU:
			if self.data_4D_lab is not None and self.data_4D_unl_clean is None: # supervised training mode
				self.bn_BU.set_runmode(0)
			elif self.data_4D_lab is not None and self.data_4D_unl_clean is not None: # semisupervised training mode
				if self.update_mean_var_with_sup:
					self.bn_BU.set_runmode(0)
				else:
					self.bn_BU.set_runmode(1)

		[self.latents_before_BN_lab, self.latents_lab, self.max_over_a_mask_lab,
		 self.max_over_t_mask_lab, self.latents_masked_lab, self.masked_mat_lab, self.output_lab, self.mask_input_lab, self.scale_s_lab,
		 self.latents_demeaned_lab, self.latents_demeaned_squared_lab] \
			= self.get_important_latents_BU(input=self.data_4D_lab, betas=betas)

		if self.is_noisy: # run bottom up for unlabeled data when noise is added
			if self.data_4D_unl is not None:
				if self.is_bn_BU:
					self.bn_BU.set_runmode(1)  # no update of means and vars
				[self.latents_before_BN, self.latents, self.max_over_a_mask, self.max_over_t_mask, self.latents_masked, self.masked_mat,
				 self.output, self.mask_input, self.scale_s, self.latents_demeaned, self.latents_demeaned_squared] \
					= self.get_important_latents_BU(input=self.data_4D_unl, betas=betas)

			if self.data_4D_unl_clean is not None:
				if self.is_bn_BU:
					self.bn_BU.set_runmode(0)  # using clean data to keep track the means and the vars
				[self.latents_before_BN_clean, self.latents_clean, self.max_over_a_mask_clean, self.max_over_t_mask_clean, self.latents_masked_clean,
				 self.masked_mat_clean, self.output_clean, self.mask_input_clean, self.scale_s_clean,
				 self.latents_demeaned_clean, self.latents_demeaned_squared_clean] \
					= self.get_important_latents_BU(input=self.data_4D_unl_clean, betas=betas)

		else: # run bottom up for unlabeled data when there is no noise
			if self.data_4D_unl is not None:
				if self.is_bn_BU:
					if self.update_mean_var_with_sup:
						self.bn_BU.set_runmode(1) # using clean data to keep track the means and vars
					else:
						self.bn_BU.set_runmode(0)  # using clean data to keep track the means and vars

				[self.latents_before_BN, self.latents, self.max_over_a_mask, self.max_over_t_mask, self.latents_masked, self.masked_mat,
				 self.output, self.mask_input, self.scale_s, self.latents_demeaned, self.latents_demeaned_squared] \
					= self.get_important_latents_BU(input=self.data_4D_unl, betas=betas)
				self.output_clean = self.output

		if self.data_4D_unl is not None:
			self.pi_t_minibatch = tf.reduce_mean(self.max_over_t_mask, axis=0)
			self.pi_a_minibatch = tf.reduce_mean(self.max_over_a_mask, axis=0)
			self.pi_ta_minibatch = tf.reduce_mean(self.masked_mat, axis=0)
			self.pi_t_new = self.momentum_pi_t * self.pi_t + (1. - self.momentum_pi_t) * self.pi_t_minibatch
			self.pi_a_new = self.momentum_pi_a*self.pi_a + (1 - self.momentum_pi_a)*self.pi_a_minibatch
			self.pi_ta_new = self.momentum_pi_ta * self.pi_ta + (1. - self.momentum_pi_ta) * self.pi_ta_minibatch
			print "hereeeee", self.mask_input.get_shape()
			if self.is_prun_synap:
				padded_mask_input, padded_shape = self.pad_images(images=tf.transpose(self.mask_input, [1, 0, 2, 3]),
																  image_shape=(self.Cin, self.Ni, self.H, self.W),
																  filter_size=[1,1,self.h, self.w],
																  border_mode=self.border_mode)
				print "hereeeee", padded_mask_input.get_shape()
				pi_synap_minibatch = tf.nn.conv2d(tf.transpose(padded_mask_input, [0,2,3,1]), tf.transpose(self.max_over_a_mask,[2,3,0,1]), strides=[1, 1, 1, 1], padding="VALID")
				pi_synap_minibatch = tf.transpose(pi_synap_minibatch,[0,3,1,2])
				# pi_synap_minibatch = tf.nn.conv2d(
				# 	data_format = "NCHW",input=padded_mask_input,
				# 	strides = [1,1,1,1],
				# 						   # filter=tf.transpose(self.max_over_a_mask,[1, 0, 2, 3]),
				# 						   filter = [self.latents_shape[2], self.latents_shape[3], self.Ni, self.K],
				# 						   padding='VALID')

				self.pi_synap_minibatch = tf.cast(tf.transpose(pi_synap_minibatch, [1, 0, 2, 3])
														 /tf.cast((self.Ni*self.latents_shape[2]*self.latents_shape[3]), tf.float32), tf.float32)

		else: # if supervised learning, compute the pi_synap from each minibatch using labeled data
			print "hohohoho"
			self.pi_t_minibatch = tf.reduce_mean(self.max_over_t_mask_lab, axis=0)
			self.pi_a_minibatch = tf.reduce_mean(self.max_over_a_mask_lab, axis=0)
			self.pi_ta_minibatch = tf.reduce_mean(self.masked_mat_lab, axis=0)
			self.pi_t_new = self.momentum_pi_t * self.pi_t + (1 - self.momentum_pi_t) * self.pi_t_minibatch
			self.pi_a_new = self.momentum_pi_a * self.pi_a + (1 - self.momentum_pi_a) * self.pi_a_minibatch
			self.pi_ta_new = self.momentum_pi_ta * self.pi_ta + (1 - self.momentum_pi_ta) * self.pi_ta_minibatch

			if self.is_prun_synap:
				padded_mask_input, padded_shape = self.pad_images(images=tf.transpose(self.mask_input_lab, [1, 0, 2, 3]),
																  image_shape=(self.Cin, self.Ni, self.H, self.W),
																  filter_size=[1,1, self.h, self.w],
																  border_mode=self.border_mode)

				pi_synap_minibatch = tf.nn.conv2d(tf.transpose(padded_mask_input, [0,2,3,1]), tf.transpose(self.max_over_a_mask,[2,3,0,1]), strides=[1, 1, 1, 1], padding="VALID")
				pi_synap_minibatch = tf.transpose(pi_synap_minibatch,[0,3,1,2])
				# pi_synap_minibatch = tf.nn.conv2d(
				# 	data_format = "NCHW",input=padded_mask_input,
				# 	strides = [1,1,1,1],
				# 							filter=self.max_over_a_mask_lab.transpose(1, 0, 2, 3),
				# 							padding='valid')

				self.pi_synap_minibatch = tf.cast(tf.transpose(pi_synap_minibatch, [1, 0, 2, 3])
										   / tf.cast(self.Ni * self.latents_shape[2] * self.latents_shape[3], tf.float32),
										   tf.float32)


	def ETopDown(self, mu_cg, mu_cg_lab):
		#
		# Reconstruct the images to compute the complete-data log-likelihood
		#
		if tf.contrib.framework.is_tensor(mu_cg) == False:
			if mu_cg != None:
				mu_cg = tf.convert_to_tensor(mu_cg, dtype = tf.float32)
			mu_cg_lab = tf.convert_to_tensor(mu_cg_lab, dtype = tf.float32)
		# self.a = tf.tile(mu_cg,multiples = [1,1,2,1])
		# Up-sample the latent presentations

		mu_cg_sh = mu_cg_lab.get_shape().as_list()
		# self.a = tf.reshape(tf.tile(tf.reshape(mu_cg,[mu_cg_sh[0],-1,1,mu_cg_sh[3]]),[1,1,2,1]),[mu_cg_sh[0],mu_cg_sh[1],-1,mu_cg_sh[3]])
		# self.a = self._repeat(mu_cg, mu_cg_sh, [1,1,2,1])
		# self.a = tf.reshape(tf.tile(tf.reshape(mu_cg,[mu_cg_sh[0],-1,1,mu_cg_sh[3]]),[1,1,2,1]),[mu_cg_sh[0],mu_cg_sh[1],-1,mu_cg_sh[3]])
		if self.pool_t_mode == 'max_t':
			if mu_cg != None:
				latents_unpooled_no_mask = self._repeat(mu_cg, mu_cg_sh,[1,1,2,1])
				self.latents_unpooled_no_mask = self._repeat1(latents_unpooled_no_mask, latents_unpooled_no_mask.get_shape().as_list(), [1,1,1,2])

			latents_unpooled_no_mask_lab = self._repeat(mu_cg_lab, mu_cg_sh,[1,1,2,1])
			self.latents_unpooled_no_mask_lab = self._repeat1(latents_unpooled_no_mask_lab, latents_unpooled_no_mask_lab.get_shape().as_list(), [1,1,1,2])
			
		elif self.pool_t_mode == 'mean_t':
			if mu_cg != None:
				latents_unpooled_no_mask = self._repeat(mu_cg, mu_cg_sh,[1,1,self.mean_pool_size[0],1])
				self.latents_unpooled_no_mask = self._repeat1(latents_unpooled_no_mask, latents_unpooled_no_mask.get_shape().as_list(), [1,1,1,self.mean_pool_size[1]])			

			latents_unpooled_no_mask_lab = self._repeat(mu_cg_lab, mu_cg_sh,[1,1,self.mean_pool_size[0],1])
			self.latents_unpooled_no_mask_lab = self._repeat1(latents_unpooled_no_mask_lab, latents_unpooled_no_mask_lab.get_shape().as_list(), [1,1,1,self.mean_pool_size[1]])			
		elif self.pool_t_mode is None:
			if mu_cg != None:
				self.latents_unpooled_no_mask = mu_cg
			self.latents_unpooled_no_mask_lab = mu_cg_lab
		else:
			raise

		# apply the a and t infered in the E-step bottom-up inference on the up-sampled intermediate image
		latents_unpooled_no_mask_shp = self.latents_unpooled_no_mask_lab.get_shape()
		masked_mat_shp = self.masked_mat_lab.get_shape()
		if latents_unpooled_no_mask_shp!=masked_mat_shp:
			print "entering"

			if latents_unpooled_no_mask_shp[2] != masked_mat_shp[2]:
				if mu_cg != None:
					self.latents_unpooled_no_mask = tf.concat([self.latents_unpooled_no_mask, tf.expand_dims(self.latents_unpooled_no_mask[:,:,-1,:],2)],2)
				self.latents_unpooled_no_mask_lab = tf.concat([self.latents_unpooled_no_mask_lab, tf.expand_dims(self.latents_unpooled_no_mask_lab[:,:,-1,:],2)],2)
			if latents_unpooled_no_mask_shp[3] != masked_mat_shp[3]:
				if mu_cg != None:
					self.latents_unpooled_no_mask = tf.concat([self.latents_unpooled_no_mask, tf.expand_dims(self.latents_unpooled_no_mask[:,:,:,-1],3)],3)
				self.latents_unpooled_no_mask_lab = tf.concat([self.latents_unpooled_no_mask_lab, tf.expand_dims(self.latents_unpooled_no_mask_lab[:,:,:,-1],3)],3)
		if mu_cg != None:
			self.latents_unpooled = self.latents_unpooled_no_mask * self.masked_mat * self.prun_mat
		self.latents_unpooled_lab = self.latents_unpooled_no_mask_lab * self.masked_mat_lab * self.prun_mat
		# print self.latents_unpooled.get_shape()
		# reconstruct/sample the image
		self.lambdas_t_deconv = tf.transpose(tf.reshape(self.lambdas_t[:, :, 0],
										 shape=(self.K, self.Cin, self.h, self.w)) * self.prun_synap_mat,[1, 0, 2, 3])
		self.lambdas_t_deconv = self.lambdas_t_deconv[:, :, ::-1, ::-1]

		if self.border_mode == 'VALID':
			if mu_cg != None:
				latents_unpooled, _ = self.pad_images(self.latents_unpooled,self.latents_unpooled.get_shape(),self.lambdas_t_deconv.get_shape(), "FULL")
				# print _
				self.data_reconstructed = tf.nn.conv2d(
					data_format = "NCHW",
					strides = [1,1,1,1],
					input= latents_unpooled,
					filter=tf.transpose(self.lambdas_t_deconv,[2,3,1,0]),
					padding='VALID'
				)
			latents_unpooled_lab, _ = self.pad_images(self.latents_unpooled_lab,self.latents_unpooled_lab.get_shape(),self.lambdas_t_deconv.get_shape(), "FULL")
			self.data_reconstructed_lab = tf.nn.conv2d(
				data_format = "NCHW",
				strides = [1,1,1,1],
				input=latents_unpooled_lab,
				filter=tf.transpose(self.lambdas_t_deconv,[2,3,1,0]),
				padding='VALID'
			)

		elif self.border_mode == 'HALF':
			if mu_cg != None:
				latents_unpooled, _ = self.pad_images(self.latents_unpooled,self.latents_unpooled.get_shape(),self.lambdas_t_deconv.get_shape(), "HALF")
				self.data_reconstructed = tf.nn.conv2d(
					data_format = "NCHW",
					strides = [1,1,1,1],
					input=latents_unpooled,
					filter=tf.transpose(self.lambdas_t_deconv,[2,3,1,0]),
					padding='VALID'
				)
			latents_unpooled_lab, _ = self.pad_images(self.latents_unpooled_lab,self.latents_unpooled_lab.get_shape(),self.lambdas_t_deconv.get_shape(), "HALF")
			self.data_reconstructed_lab = tf.nn.conv2d(
				data_format = "NCHW",
				strides = [1,1,1,1],
				input=latents_unpooled_lab,
				filter=tf.transpose(self.lambdas_t_deconv,[2,3,1,0]),
				padding='VALID'
			)
		else:
			if mu_cg != None:
				self.data_reconstructed = tf.nn.conv2d(
					data_format = "NCHW",
					strides = [1,1,1,1],
					input=self.latents_unpooled,
					filter=tf.transpose(self.lambdas_t_deconv,[2,3,1,0]),
					padding='VALID'
				)
			self.data_reconstructed_lab = tf.nn.conv2d(
				data_format = "NCHW",
				strides = [1,1,1,1],
				input=self.latents_unpooled_lab,
				filter=tf.transpose(self.lambdas_t_deconv,[2,3,1,0]),
				padding='VALID'
			)

		# compute reconstruction error
		if mu_cg != None:
			self.reconstruction_error = tf.reduce_mean((self.data_4D_unl_clean - self.data_reconstructed) ** 2)
		self.reconstruction_error_lab = tf.reduce_mean((self.data_4D_lab - self.data_reconstructed_lab) ** 2)
		tf.summary.scalar("reconstruction_error", self.reconstruction_error_lab)

	def pad_images(self, images, image_shape, filter_size, border_mode):
		"""
		pad image with the given pad_size
		"""
		# Allocate space for padded images.
		
		if border_mode == 'VALID':
			x_padded = images
			padded_shape = image_shape
		else:
			if border_mode == 'HALF':
				h_pad = filter_size[2] // 2
				w_pad = filter_size[3] // 2
			elif border_mode == 'FULL':
				h_pad = filter_size[2] - 1
				w_pad = filter_size[3] - 1

			s = image_shape.as_list()
			# padded_shape = (s[0], s[1], s[2] + 2*h_pad, s[3] + 2*w_pad)
			row_pad1 = tf.zeros([s[0], s[1], h_pad, s[3]])
			row_pad2 = tf.zeros([s[0], s[1], h_pad, s[3]])
			print h_pad
			x_padded_r = tf.concat([row_pad2, images, row_pad1],2)
			col_pad1 = tf.zeros([s[0], s[1], s[2] + 2*int(h_pad), w_pad])
			col_pad2 = tf.zeros([s[0], s[1], s[2] + 2*int(h_pad), w_pad])
			
			x_padded = tf.concat([col_pad1, x_padded_r, col_pad2], 3)
			# x_padded = tf.zeros(padded_shape)
			padded_shape = [s[0], s[1], s[2]+2*int(h_pad), s[3]+2*int(w_pad)]
			# # Copy the original image to the central part.
			# x_padded = T.set_subtensor(
			# 	x_padded[:, :, h_pad:s[2]+h_pad, w_pad:s[3]+w_pad],
			# 	images,
			# )

		return x_padded, padded_shape
	def _repeat(self, inp, inp_shape, multiples):
		return tf.reshape(tf.tile(tf.reshape(inp,[inp_shape[0],-1,1,inp_shape[3]]),multiples),[inp_shape[0],inp_shape[1],-1,inp_shape[3]])

	def _repeat1(self, inp, inp_shape, multiples):
		return tf.reshape(tf.tile(tf.reshape(inp,[inp_shape[0],inp_shape[1],-1,1]),multiples),[inp_shape[0],inp_shape[1],inp_shape[2],-1])
