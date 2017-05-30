import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# from model import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from test import *

# print model.N_layers
class Train():
	"""This class implemnents the training rule routine"""
	def __init__(self, model = None, args = None):
		if model == None:
			self.CreateModel(args)
		else:
			self.model = model
		self.load_data()


	def CreateModel(self, args = None):
		batch_size =  100
		Cin = 1
		W = 28
		H = 28
		seed = 23

		model = Model(batch_size, Cin, W, H, seed)
		model.add(noise_weight = 0.0, noise_std = 0.01, K = 32, W = 28, H = 28, M = 1, w = 3, h = 3, Ni = 100, Cin = 1, border_mode = "VALID")
		model.add(noise_weight = 0.0, noise_std = 0.01, K = 64, W = 64, H = 64, M = 1, w = 3, h = 3, Ni = 100, Cin = 1, border_mode = "VALID")
		model.add(noise_weight = 0.0, noise_std = 0.01, K = 100, W = 64, H = 64, M = 1, w = 3, h = 3, Ni = 100, Cin = 1, border_mode = "VALID")
		print model.layer.output_lab.get_shape().as_list()

	def load_data(self, name = "MNIST"):
		self.mnist = input_data.read_data_sets('MNIST_data')

	def reshape_data(self, data, to_shape = (1, 28, 28)):
		data_sh = data.shape
		return data.reshape((data_sh[0], to_shape[0], to_shape[1], to_shape[2]))

	def train(self):
		epochs = 1
		with  tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			print "starting the training"
			for epoch in range(epochs):
				batch_no = 0
				# print
				while True:

					try:
						train = self.mnist.train.next_batch(100)
						# print train[1]
						# break
						feed_dict = {
									self.model.x_lab : self.reshape_data(train[0]),
									self.model.y_lab : train[1],
									self.model.lr : 0.01,
									self.model.momentum_bn : 0.99,
									self.model.is_train : 1
									# self.model.x_unl : self.reshape_data(train[0]),
									# self.model.x_clean : self.reshape_data(train[0])
						}
						print "here"
						_, train_loss = sess.run([self.model.train_ops, self.model.cost], feed_dict = feed_dict)			
						print "gogo"
						# err = sess.run(self.model.cost, feed_dict={self.model.x_lab: self.reshape_data(self.mnist.test.images), self.model.y_lab : self.mnist.test.labels})
																	# self.model.x_unl : None, self.model.x_clean : None											
# self.																	})
						print train_loss
						print "train error after training %s batches is %s".format(batch_no, train_loss)
						batch_no += 1

					except Exception as e:
						raise e	
				


# a = Train()
a = Train(model)

print type(a.mnist)
a.train()		
# a.CreateModel()
