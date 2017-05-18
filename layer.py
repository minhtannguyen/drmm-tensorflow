"""
This file implements the a single layer of a model
"""

__author__ = 'rishabgoel'
"""
Based on theano code minhtannguyen
"""

import numpy as np
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
		`K`:           Number of the rendering matrices lambdas
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
		`lambdas_val_init`: Initial values for rendering matrices
		`amps_val_init`: Initial values for priors
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
				 lambdas_val_init=None, amps_val_init=None,
				 gamma_val_init=None, beta_val_init=None,
				 prun_mat_init=None, prun_synap_mat_init=None,
				 mean_bn_init=None, var_bn_init=None,
				 pool_t_mode='max_t', border_mode='valid', nonlin='relu', pool_a_mode='relu',
				 mean_pool_size=(2, 2),
				 max_condition_number=1.e3,
				 weight_init="xavier",
				 is_noisy=False,
				 is_bn_BU=False,
				 epsilon=1e-10,
				 momentum_pi_t=0.99,
				 momentum_pi_a=0.99,
				 momentum_pi_ta=0.99,
				 momentum_pi_synap=0.99,
				 is_prun=False,
				 is_prun_synap=False,
				 is_dn=False,
				 sigma_dn_init=None,
				 b_dn_init=None,
				 alpha_dn_init=None,
				 update_mean_var_with_sup=False):

		self.K = K  # number of lambdas/filters
		self.M = M  # latent dimensionality. Set to 1 for our model
		self.data_4D_lab = data_4D_lab  # labeled data
		self.data_4D_unl = data_4D_unl  # unlabeled data
		self.data_4D_unl_clean = data_4D_unl_clean  # unlabeled data used for the clean path
		self.noise_weight = noise_weight  # control how much noise we want to add to the model. Set to 0 for our model
		self.noise_std = noise_std  # the standard deviation of the noise
		self.Ni = Ni  # no. of labeled examples = no. of unlabeled examples
		self.w = w  # width of filters
		self.h = h  # height of filters
		self.Cin = Cin  # number of channels in the image
		self.D = self.h * self.w * self.Cin  # patch size
		self.W = W  # width of image
		self.H = H  # height of image
		if border_mode == 'valid':  # convolution mode. Output size is smaller than input size
			self.Np = (self.H - self.h + 1) * (self.W - self.w + 1)  # no. of patches per image
			self.latents_shape = (self.Ni, self.K, self.H - self.h + 1, self.W - self.w + 1)
		elif border_mode == 'half':  # convolution mode. Output size is the same as input size
			self.Np = self.H * self.W  # no. of patches per image
			self.latents_shape = (self.Ni, self.K, self.H, self.W)
		elif border_mode == 'full':  # # convolution mode. Output size is greater than input size
			self.Np = (self.H + self.h - 1) * (self.W + self.w - 1)  # no. of patches per image
			self.latents_shape = (self.Ni, self.K, self.H + self.h - 1, self.W + self.w - 1)
		else:
			raise

		self.N = self.Ni * self.Np  # total no. of patches and total no. of hidden units
		self.mean_pool_size = mean_pool_size  # size of the mean pooling layer before the softmax regression

		self.lambdas_val_init = lambdas_val_init  # Initial values for rendering matrices
		self.amps_val_init = amps_val_init  # Initial values for priors
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
		self.pool_a_mode = pool_a_mode # {'relu', None} we are not using this. we are using nonlin instead
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
		self._inialize()

	def _initialize(self):
		#
		# initialize the model parameters.
		# all parameters involved in the training are collected in self.params for the gradient descent step
		#

		# set up a random number generator
		self.srng = RandomStreams()
		self.srng.seed(np.random.randint(2 ** 15))

		# add noise to the input if is_noisy
		if self.is_noisy:
			if self.data_4D_unl is not None:
				self.data_4D_unl = self.data_4D_unl + \
								   self.noise_weight * self.srng.normal(size=(self.Ni, self.Cin, self.H, self.W),
																		avg=0.0, std=self.noise_std)
			if self.data_4D_lab is not None:
				self.data_4D_lab = self.data_4D_lab + \
								   self.noise_weight * self.srng.normal(size=(self.Ni, self.Cin, self.H, self.W),
																		avg=0.0, std=self.noise_std)

		# initialize amps
		# amps is class prior  e.g. cat, dog probabilities
		if self.amps_val_init is None:
			amps_val = np.random.rand(self.K)
			amps_val /= np.sum(amps_val)

		else:
			amps_val = self.amps_val_init

		self.amps = tf.Variable(tf.convert_to_tensor(amps_val), dtype = tf.float32, name = "amps")
		# initialize t and a priors
		self.pi_t = tf.Variable(tf.ones([self.latents_shape[1:]]), name = "pi_t")
		self.pi_a = tf.Variable(tf.ones([self.latents_shape[1:]]), name = "pi_a")
		self.pi_a_old = tf.Variable(tf.ones([self.latents_shape[1:]]), name = "pi_a_old")
		self.pi_ta = tf.Variable(tf.ones([self.latents_shape[1:]]), name = "pi_ta")
		

		# initialize the pruning masking matrices
		if self.prun_mat_init is None:
			self.prun_mat = tf.Variable(tf.ones([self.latents_shape[1:]]), name = "prun_mat")
		else:
			self.prun_mat = tf.Variable(tf.convert_to_tensor(np.asarray(self.prun_mat_init), dtype = tf.float32, name = "prun_mat"))
			
		if self.prun_synap_mat_init is None:
			self.prun_synap_mat = tf.Variable(tf.ones([self.K, self.Cin, self.h, self.w]), dtype=tf.float32,name='prun_synap_mat', borrow=True)
		else:
			self.prun_synap_mat = tf.Variable(tf.convert_to_tensor(np.asarray(self.prun_synap_mat_init)), dtype=tf.float32, name='prun_mat')

		# initialize synapse prior (its probability to be ON or OFF)
		if self.is_prun_synap:
			self.pi_synap = tf.Variable(tf.ones([self.K, self.Cin, self.h, self.w], dtype = tf.float32), name = 'pi_synap')
			self.pi_synap_old = tf.Variable(tf.ones([self.K, self.Cin, self.h, self.w], dtype = tf.float32), name = 'pi_synap_old')
			
		# pi_t_final and pi_a_final are used after training for sampling
		self.pi_t_final = tf.Variable(tf.ones([self.latents_shape[1:]], dtype = tf.float32), name = "pi_t_final")
		self.pi_a_final = tf.Variable(tf.ones([self.latents_shape[1:]], dtype = tf.float32), name = "pi_a_final")
		
		# initialize the lambdas
		# if initial values for lambdas are not provided, randomly initialize lambdas
		initialised = 0
		if self.lambdas_val_init is None:
			if self.weight_init == "xavier":
				initialised = 1
				self.lambdas = tf.Variable(name = "lambdas", shape = [self.K, self.D, self.M],  initializer = tf.contrib.layers.xavier_initializer())
			else:
				lambdas_value = np.random.randn(self.K, self.D, self.M) / \
								np.sqrt(self.max_condition_number)
		else:
			lambdas_value = self.lambdas_val_init
		if initialised == 0:
			self.lambdas = tf.Variable(tf.convert_to_tensor(lambdas_value, dtype=tf.float32), name='lambdas')

		# Initialize BatchNorm
		if self.is_bn_BU:
			self.bn_BU = BatchNormalization(insize=self.K, mode=1, momentum=self.momentum_bn, is_train=self.is_train,
											epsilon=self.epsilon,
											gamma_val_init=self.gamma_val_init, beta_val_init=self.beta_val_init,
											mean_init=self.mean_bn_init, var_init=self.var_bn_init)

			self.params = [self.lambdas, self.bn_BU.gamma, self.bn_BU.beta]
		else:
			self.params = [self.lambdas, ]

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


	def EBottomUp(self, args):
		
	def ETopDown(self, args):
