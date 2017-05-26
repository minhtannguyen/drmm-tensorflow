"""
This file builds the computational graph of the model
"""
__author__ = 'rishabgoel'
import matplotlib as mpl

mpl.use('Agg')
from pylab import *

import copy

import time
import tensorflow as tf
import numpy as np

from layer import *
from utils import *


class Model():
	"""This class implememts the creating a deep RMM"""
	'''
	`batch_size`:   size of minibatches used in each iteration
	`Cin`: The number of channels of the input
	`W`: The width of the input
	`H`: The height of the input
	`seed`: random seed
	`param_dir`: dir that saved the trained DRMM. Used in transfer learning or just to continue the training that stopped
				 for some reason
	`reconst_weights`: used to weigh reconstruction cost at each layer in the DRMM
	`xavier_init`: {True, False} use Xavier initialization or not
	`grad_min, grad_max`: clip the gradients
	`init_params`: {True, False} if True, load parameters from a trained DRMM
	`noise_std`: standard deviation of the noise added to the model
	`noise_weights`: control how much noise we want to add to the model. We always noise_weight to 0
	`is_bn_BU`: {True, False} use batch normalization in the Bottom-Up or not
	`nonlin`: {relu, abs} the nonlinearity to use in the Bottom-Up
	`method`: {SGD, adam} training method
	`KL_coef`: weight of the KL divergence cost
	`sign_cost_weight`: weight of the sign cost
	`is_prun`: {True, False} do neuron pruning or not
	`is_prun_synap`: {True, False} do synap pruning or not
	`is_prun_songhan`: {True, False} do synap pruning using Song Han's weight thresholding method as in Deep Compression
	`is_finetune_after_prun`: {True, False} finetune the model after the pruning finishes or not
	`is_dn`: {True, False} apply divisive normalization (DivNorm) or not
	`update_mean_var_with_sup`: {True, False} if True, only use data_4D_lab to update the mean and var in BatchNorm
	`cost_mode`: {'sup', 'unsup', 'both'} use labeled data, unlabeled data, or both to compute the costs
	`is_sup`: {True, False} do supervised learning or not
	'''
	# TODO: factor out common init code from all models
	def __init__(self, batch_size, seed, param_dir=[],
				 reconst_weights=0.0, xavier_init=False, grad_min=-np.inf, grad_max=np.inf,
				 init_params=False, noise_std=0.45, noise_weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
				 is_bn_BU=False, nonlin='relu',
				 method='SGD', KL_coef=0.0, sign_cost_weight=0.0,
				 is_prun=False, is_prun_synap=False, is_prun_songhan=False, is_finetune_after_prun=False,
				 param_dir_for_prun_mat=[], is_dn=False,
				 update_mean_var_with_sup=False,
				 cost_mode='unsup',num_class = 200,
				 is_sup=False):

		self.num_class = num_class # number of object classes in the BlenderRender training set
		self.noise_std = noise_std
		self.noise_weights = noise_weights
		self.reconst_weights = reconst_weights
		self.sign_cost_weight = sign_cost_weight
		self.is_bn_BU = is_bn_BU
		self.batch_size = batch_size
		self.grad_min = grad_min
		self.grad_max = grad_max
		self.nonlin = nonlin
		self.KL_coef = KL_coef
		self.method = method
		self.is_prun = is_prun
		self.is_prun_synap = is_prun_synap
		self.is_prun_songhan = is_prun_songhan
		self.is_dn = is_dn
		self.update_mean_var_with_sup = update_mean_var_with_sup
		self.cost_mode = cost_mode
		self.is_sup = is_sup


		W_init = None
		b_init = None

		self.seed = seed
		np.random.seed(self.seed)

		
		# placeholders for the object labels
		self.y_lab = tf.placeholder(tf.int64,[batch_size], 'y_lab')

		self.layers = []
		# self.layer_names = []
		self.N_layers = 0
		# self.survive_thres = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		self.survive_thres = []
		self.prun_threshold = []
		self.prun_weights = []
		# placeholders for other hyper-paramaters that changes during the training
		self.lr = tf.placeholder(tf.float32, [], name = "l_r")
		# self.prun_threshold = tf.placeholder(tf.float32, [None], name = "prun_threshold")
		# self.prun_weights = tf.placeholder(tf.float32, [None], name = prun_weights)
		self.is_train = tf.placeholder(tf.float32, [], 'is_train')
		self.momentum_bn = tf.placeholder(tf.float32, [], 'momentum_bn')
		self.optimizer = tf.train.AdamOptimizer()

		self._load_pretrained()

	def add(self, noise_weight, noise_std, K, M, W, H, w, h, Cin, Ni,
			lambdas_t_val_init=None, gamma_val_init=None, beta_val_init=None,prun_mat_init=None, prun_synap_mat_init=None,mean_bn_init=None, var_bn_init=None,pool_t_mode='max_t', 
			border_mode='VALID', nonlin='abs', mean_pool_size=[2, 2],max_condition_number=1.e3,weight_init="xavier",is_noisy=False, is_bn_BU=True,
			epsilon=1e-10,momentum_pi_t=0.99,momentum_pi_a=0.99,momentum_pi_ta=0.99,momentum_pi_synap=0.99, is_prun=True, is_prun_synap=True,
			is_dn=True,sigma_dn_init=None,b_dn_init=None,alpha_dn_init=None,update_mean_var_with_sup=True, survive_thres = 0.0, name = "conv_layer"):
		# Building the forward model
		with tf.variable_scope("drmm_layer_no_" + str(self.N_layers)):
			if self.N_layers > 0:
				shapes = self.layer.output_lab.get_shape().as_list()
				# print shapes,"fffffff\n\n\n\n\n", self.N_layers
				layer = Layer(
					data_4D_lab=self.layer.output_lab,
					data_4D_unl=self.layer.output,
					data_4D_unl_clean=self.layer.output_clean,
					noise_weight=noise_weight,
					noise_std=noise_std,
					is_train=self.is_train, momentum_bn=self.momentum_bn,
					K=K, M=M, W=shapes[2], H=shapes[3],
					w=w, h=h, Cin=shapes[1], Ni=self.batch_size,
					lambdas_t_val_init=lambdas_t_val_init,
					gamma_val_init=gamma_val_init, beta_val_init=beta_val_init,
					prun_mat_init=prun_mat_init, prun_synap_mat_init=prun_synap_mat_init,
					mean_bn_init=mean_bn_init, var_bn_init=var_bn_init,
					sigma_dn_init=sigma_dn_init, b_dn_init=b_dn_init, alpha_dn_init=alpha_dn_init,
					update_mean_var_with_sup=self.update_mean_var_with_sup,
					pool_t_mode=pool_t_mode,
					border_mode=border_mode,
					max_condition_number=1.e3, weight_init=weight_init,
					is_noisy=is_noisy, is_bn_BU=self.is_bn_BU,
					nonlin=self.nonlin,
					is_prun=self.is_prun,
					is_prun_synap=self.is_prun_synap,
					is_dn=self.is_dn,
					name = "drmm_layer_no_" + str(self.N_layers)
				)
			else:
				#create x variables for the first time
				self.x_lab = tf.placeholder(tf.float32,[self.batch_size, Cin, H, W],'x_lab')
				self.x_unl = tf.placeholder(tf.float32,[self.batch_size, Cin, H, W],'x_unl')
				self.x_clean = tf.placeholder(tf.float32,[self.batch_size, Cin, H, W],'x_unl')

				layer = Layer(
					data_4D_lab=self.x_lab,
					data_4D_unl=self.x_unl,
					data_4D_unl_clean=self.x_clean,
					noise_weight=noise_weight,
					noise_std=noise_std,
					is_train=self.is_train, momentum_bn=self.momentum_bn,
					K=K, M=M, W=W, H=H,
					w=w, h=h, Cin=Cin, Ni=self.batch_size,
					lambdas_t_val_init=lambdas_t_val_init,
					gamma_val_init=gamma_val_init, beta_val_init=beta_val_init,
					prun_mat_init=prun_mat_init, prun_synap_mat_init=prun_synap_mat_init,
					mean_bn_init=mean_bn_init, var_bn_init=var_bn_init,
					sigma_dn_init=sigma_dn_init, b_dn_init=b_dn_init, alpha_dn_init=alpha_dn_init,
					update_mean_var_with_sup=self.update_mean_var_with_sup,
					pool_t_mode=pool_t_mode,
					border_mode=border_mode,
					max_condition_number=1.e3, weight_init=weight_init,
					is_noisy=is_noisy, is_bn_BU=self.is_bn_BU,
					nonlin=self.nonlin,
					is_prun=self.is_prun,
					is_prun_synap=self.is_prun_synap,
					is_dn=self.is_dn,
					name = "drmm_layer_no_1"
				)

			layer.EBottomUp()
			self.layers.append(layer)
			self.layer = layer
			self.N_layers += 1
			self.survive_thres.append(survive_thres)

	def Compile(self):
		# regression layer for softmax
		print self.layers[-1].D,"fjdfdsjfsddsjkhdsjkfksd", self.num_class
		self.RegressionInSoftmax = HiddenLayerInSoftmax(input_lab=tf.reshape(self.layer.output_lab,[self.batch_size,-1]),
														input_unl=tf.reshape(self.layer.output,[self.batch_size,-1]),
														input_clean=tf.reshape(self.layer.output_clean,[self.batch_size,-1]), 
														n_in = self.layers[-1].K, n_out=self.num_class, W_init=None, b_init=None)

		softmax_input_lab = self.RegressionInSoftmax.output_lab
		softmax_input = self.RegressionInSoftmax.output
		softmax_input_clean = self.RegressionInSoftmax.output_clean
		print softmax_input_lab.get_shape(),"sdfsdfds"
		# softmax nonlinearity for object recognition
		self.softmax_layer_nonlin = SoftmaxNonlinearity(input_lab=softmax_input_lab, input_unl=softmax_input,input_clean=softmax_input_clean)
		print self.softmax_layer_nonlin.y_pred_lab.get_shape()
		# build Top-Down pass
		self._Build_TopDown_End_to_End()

		# build the cost function for the model
		self.BuildCost()

		# build update rules for the model
		# self.Build_Update_Rule(method=self.method)


	def _Build_TopDown_End_to_End(self):
		print tf.one_hot(self.softmax_layer_nonlin.y_pred, self.num_class).get_shape()
		self.top_output = tf.matmul(tf.one_hot(self.softmax_layer_nonlin.y_pred, self.num_class),
						tf.transpose(self.RegressionInSoftmax.W))
		self.top_output_lab = tf.matmul(tf.one_hot(self.softmax_layer_nonlin.y_pred_lab, self.num_class),
								tf.transpose(self.RegressionInSoftmax.W))
		print self.top_output.get_shape()
		self.top_output = tf.reshape(self.top_output, [self.batch_size, -1, 1,1])
		self.top_output_lab = tf.reshape(self.top_output_lab, [self.batch_size, -1, 1,1])

		self.layers[-1].ETopDown(mu_cg=self.top_output, mu_cg_lab=self.top_output_lab)

		for i in xrange(1, self.N_layers):
			self.layers[self.N_layers - i - 1].ETopDown(mu_cg=self.layers[self.N_layers - i].data_reconstructed,
														   mu_cg_lab=self.layers[self.N_layers - i].data_reconstructed_lab)
	

	
	def BuildCost(self):
		'''
		Build the costs which are minimized during training
		:return:
		'''
		# compute the classification error
		self.classification_error = self.softmax_layer_nonlin.errors(self.y_lab)
		# print "classification_error_got"
		# supervised learning cost is the cross-entropy
		self.supervised_cost = self.softmax_layer_nonlin.negative_log_likelihood(tf.one_hot(self.y_lab, self.num_class))

		if self.is_sup: # is sup, use only the cross-entropy cost
			self.cost = self.supervised_cost
		else: # if semi-sup, use cross-entropy cost + reconstruction cost + KL penalty + NN penalty
			# Build the NN penalty
			self.sign_cost_unl = 0.0
			self.sign_cost_lab = 0.0
			for i in xrange(self.N_layers):
				self.sign_cost_unl += tf.reduce_mean(tf.nn.relu(-self.layers[i].latents_unpooled_no_mask) ** 2)
				self.sign_cost_lab += tf.reduce_mean(tf.nn.relu(-self.layers[i].latents_unpooled_no_mask_lab) ** 2)


			# Build the reconstruction cost
			self.unsupervised_cost_unl = tf.reduce_mean((self.layers[-1].data_4D_unl_clean - self.layers[-1].data_reconstructed) ** 2)
			self.unsupervised_cost_lab = tf.reduce_mean(
				(self.layers[-1].data_4D_lab - self.layers[-1].data_reconstructed_lab) ** 2)

			# KL penalty
			self.KLD_cost_unl = -tf.reduce_mean(tf.reduce_sum(tf.log(np.float32(self.num_class) * self.softmax_layer_nonlin.gammas + 1e-8)
										 * self.softmax_layer_nonlin.gammas, axis=1))

			self.KLD_cost_lab = -tf.reduce_mean(tf.reduce_sum(tf.log(np.float32(self.num_class) * self.softmax_layer_nonlin.gammas_lab + 1e-8)
										  * self.softmax_layer_nonlin.gammas_lab, axis=1))

			# compute the costs using both labeled data and unlabeled data, only unlabeled data, or only labeled data
			if self.cost_mode == 'both':
				self.sign_cost = 0.5 * self.sign_cost_unl + 0.5 * self.sign_cost_lab
				self.KLD_cost = 0.5 * self.KLD_cost_unl + 0.5 * self.KLD_cost_lab
				self.unsupervised_cost = 0.5 * self.unsupervised_cost_unl + 0.5 * self.unsupervised_cost_lab
			elif self.cost_mode == 'unsup':
				self.sign_cost = self.sign_cost_unl
				self.KLD_cost = self.KLD_cost_unl
				self.unsupervised_cost = self.unsupervised_cost_unl
			else:
				self.sign_cost = self.sign_cost_lab
				self.KLD_cost = self.KLD_cost_lab
				self.unsupervised_cost = self.unsupervised_cost_lab

			# Total cost
			self.cost = self.supervised_cost \
						+ self.reconst_weights * self.unsupervised_cost \
						+ self.KL_coef * self.KLD_cost \
						+ self.sign_cost_weight * self.sign_cost		


	def Optimize(self, optimizer = "adam"):
		self.optimizer.minimize(self.cost)

	def _load_pretrained(self):
		pass

	def __errors(self, y_lab):
		return


# class Model(DRMM):
# 	"""
# 	This class integerates the model together
# 	"""
# 	def __init__(self, arg):
# 		super(Model, self).__init__()
# 		self.arg = arg
# 	"""
# 	If new model used then update _BuildBottomUp
# 	Currently, it will be same the DRMM _BuildBottomUp
# 	"""
# 	def _BuildBottomUp(self, ):
# 		pass

# 	def BuildModel(self,  ):
# 		pass


# if __name__ == "__main__":
# 	a = 