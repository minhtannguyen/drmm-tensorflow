import argparse

import numpy as np
import tensorflow as tf


def main():
	"""
	This function sets the global variables
	"""
	parser = argparse.ArgumentParser(
									formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	
	parser.add_argument('--model_name', type=str, default='imagenet',
						help='name of the model')
	parser.add_argument('--data_mode', type=str, default='all',
						help='use both labelled and unlabelled data')
	parser.add_argument('--id_bn_BU', type=int, default=0,
						help='use BatchNorm in the Bottom Up (BU)')
	parser.add_argument('--lr_init', type=float, default=0.001,
						help='initial learning rate')
	parser.add_argument('--lr_final', type=float, default=0.000001,
						help='final learning rate')
	parser.add_argument('--max_epochs', type=int, default=500,
						help='maximum number of epochs')
	parser.add_argument('--noise_std', type=float, default=0.1,
						help='standard deviation of the noise')
	parser.add_argument('--', type=int, default=1,
						help='')
		


	parser.add_argument('--is_dn', type=int, default=1,
						help='USe DivNorm')
	parser.add_argument('--neuron_prun', type=int, default=0,
						help='Use neuroon pruning or not')
	parser.add_argument('--songhan_prun', type=int, default=0,
						help='use synaptic pruning using weight thresholding')
	parser.add_argument('--synap_prun', type=int, default=1,
						help='Use synapptic pruning')
	parser.add_argument('--param_file', type=str, default=None,
						help='file containing the description of the layers like channels, filter size etc')
	parser.add_argument('--init', type=int, default=1,
						help='use specific param values to initialize the model')
	parser.add_argument('--end_prun', type=int, default=1,
						help='do prun in end')
	parser.add_argument('--sup', type=int, default=0,
						help='do supervised learning or not')
	parser.add_argument('--optimizer', type=str, default='rmsprop',
						help='name of optimizer to use') # {SGD, adam, rmsprop} training method
	parser.add_argument('--activation', type=str, default='relu',
						help='Non linearity for the hidden layers') # choose the non-linearity for the BU: relu, abs, tanh,...
	parser.add_argument('--num_epochs', type=int, default=10,
						help='Number of epochs for training')
	parser.add_argument('--batch_size', type=int, default=100,
						help='size of mini-batch')
	parser.add_argument('--cost_mode', type=str, default='unsup',
						help='use labeled data, unlabeled data, or both to compute the costs') # {'sup', 'unsup', 'both'}
	parser.add_argument('--kl', type=float, default=0.5,
						help='coefficient of kl divergence in the cost function')
	parser.add_argument('--cw', type=float, default=0.5,
						help='weight of sign cost')
	parser.add_argument('--grad_min', type=float, default=0.5,
						help='min value of gradient')
	parser.add_argument('--grad_max', type=float, default=0.5,
						help='max value of gradient')
	parser.add_argument('--initialisation', type=str, default='xavier',
						help='kind of initialisation for the weights')
	parser.add_argument('--epoch_nlll', type=int, default=10,
						help='plot negative log-likelihood at each epoch')
	parser.add_argument('--monitor_batch', type=int, default=10,
						help='use this number of batches to monitor the training')
	parser.add_argument('--mean_update', type=int, default=0,
						help='if True, only use data_4D_lab to update the mean and var in BatchNorm')
	parser.add_argument('--sup', type=int, default=0,
						help='do supervised learning or not')
	parser.add_argument('--plot_full', type=int, default=0,
						help='if True, enter full-plot mode') # {True, False} if True, enter full-plot mode
	parser.add_argument('--fine_tune_prun', type=int, default=1,
						help='?????????????')
	parser.add_argument('--Nlabelled', type=int, default=43200,
						help='number of labeled examples used during training 8680')

	args = parser.parse_args()
	train(args)

def train(args):
	"""
	This function performs the training
	"""
	print("arg dict is: ", args)



main()