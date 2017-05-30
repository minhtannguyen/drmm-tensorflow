import theano
import theano.tensor as T
import tensorflow as tf
import numpy as np
from layer import *
from utils import *
from model import *
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def reshape_data(data, to_shape = (1, 28, 28)):
	data_sh = data.shape
	return data.reshape((data_sh[0], to_shape[0], to_shape[1], to_shape[2]))
mnist = input_data.read_data_sets('MNIST_data')
# def change_shape(self, vec, input_shape):
#     return T.repeat(vec, input_shape[2] * input_shape[3]).reshape(
#         (input_shape[1], input_shape[2], input_shape[3]))

# v = 2.0*np.ones(10,3,8,8)
# # print v
# a = 2.0*tf.ones([10,3,8,8])
# # b = T.shared(value = v, borrow = True)
# c = T.scalar("fdfd")
# d = T.mean(v, axis = (0,2,3))
# e = T.var(v, axis = (0,2,3))
# f = theano.function([c], d)
# print f(1)
# now_mean, now_var = tf.nn.moments(a, axes=[0, 2, 3])
# with tf.Session() as sess:
# 	print sess.run(d, feed_dict = {})


lab_img = np.random.rand(100, 1, 64, 64)
# unlab_img_np = np.random.rand(100, 3, 9, 8)
# unlab_img_clean_np = np.random.rand(100, 3, 9, 8)

# noise_wt = 0.0
# noise_std = 0.01
# K = 8
# W = 8
# H = 9
# M = 1
# w = 3
# h = 3
# Cin = 3
# Ni = 100
# momentum_bn = 0.99
# is_train = 1
y = np.random.randint(low = 0, high = 35, size = 100)

# lab_img = tf.placeholder(tf.float32, [100,3,9,8])
# unlab_img = tf.placeholder(tf.float32, [100,3,9,8])
# unlab_img_clean = tf.placeholder(tf.float32, [100,3,9,8])
# tf_layer = Layer(lab_img, unlab_img, unlab_img_clean, noise_wt, noise_std, K, M, W, H, w, h, Cin, Ni, momentum_bn, is_train)
# tf_layer.EBottomUp()
# print tf_layer.output_lab.get_shape(),"hore"

# def get_important_latents_BU_th(self, input, betas):
# 	# if self.border_mode != 'FULL':
# 	latents_before_BN = tf.nn.conv2d(tf.transpose(input, [0,2,3,1]), betas, strides=[1, 1, 1, 1], padding=self.border_mode)
# 	latents_before_BN = tf.transpose(latents_before_BN, [0, 3, 1, 2])

# def get_important_latents_BU_tf(self, input, betas):

# 	self.betas111 = betas
# 	self.i = input
# 	latents_before_BN = conv2d(
# 		input=input,
# 		filters=betas,
# 		filter_shape=(self.K, self.Cin, self.h, self.w),
# 		input_shape=(self.Ni, self.Cin, self.H, self.W),
# 		filter_flip=False,
# 		border_mode=self.border_mode
# 	)




# RegressionInSoftmax = HiddenLayerInSoftmax(input_lab=tf.reshape(tf_layer.output_lab,[Ni,-1]),input_unl=tf.reshape(tf_layer.output,[Ni,-1]),input_clean=tf.reshape(tf_layer.output_clean,[Ni,-1]),n_in=72, n_out=8,W_init=None, b_init=None)
# softmax_input_lab = RegressionInSoftmax.output_lab
# softmax_input = RegressionInSoftmax.output
# softmax_input_clean = RegressionInSoftmax.output_clean
# softmax_layer_nonlin = SoftmaxNonlinearity(input_lab=softmax_input_lab, input_unl=softmax_input,input_clean=softmax_input_clean)
# ss= softmax_layer_nonlin.y_pred_lab


# lab_img = tf.placeholder(tf.float32, [100,3,9,8])
# a = tf.convert_to_tensor(lab_img, dtype = tf.float32)
# b,c = pad_images(a, a.get_shape(), [1,8,3,3], "SAME")
# b = tf.convert_to_tensor(b, dtype = tf.float32)
# c = tf.convert_to_tensor(c, dtype = tf.float32)
# out = 
# # tf_layer.ETopDown()

# batch_size =  100
# Cin = 3
# W = 8
# H = 9
# seed = 23


# model = Model(batch_size, Cin, W, H, seed)
# model.add(noise_weight = 0.0, noise_std = 0.01, K = 96, W = 64, H = 64, M = 1, w = 3, h = 3, Ni = 100, Cin = 1, border_mode = "HALF")
# model.add(noise_weight = 0.0, noise_std = 0.01, K = 96, W = 64, H = 64, M = 1, w = 3, h = 3, Ni = 100, Cin = 1, border_mode = "FULL", pool_t_mode = None)
# model.add(noise_weight = 0.0, noise_std = 0.01, K = 96, W = 64, H = 64, M = 1, w = 3, h = 3, Ni = 100, Cin = 1, border_mode = "FULL")
# model.add(noise_weight = 0.0, noise_std = 0.01, K = 192, W = 64, H = 64, M = 1, w = 3, h = 3, Ni = 100, Cin = 1, border_mode = "VALID", pool_t_mode = None)
# model.add(noise_weight = 0.0, noise_std = 0.01, K = 192, W = 64, H = 64, M = 1, w = 3, h = 3, Ni = 100, Cin = 1, border_mode = "FULL", pool_t_mode = None)
# model.add(noise_weight = 0.0, noise_std = 0.01, K = 192, W = 64, H = 64, M = 1, w = 3, h = 3, Ni = 100, Cin = 1, border_mode = "VALID")
# model.add(noise_weight = 0.0, noise_std = 0.01, K = 192, W = 64, H = 64, M = 1, w = 3, h = 3, Ni = 100, Cin = 1, border_mode = "VALID", pool_t_mode = None)
# model.add(noise_weight = 0.0, noise_std = 0.01, K = 192, W = 64, H = 64, M = 1, w = 1, h = 1, Ni = 100, Cin = 1, border_mode = "VALID")
# model.add(noise_weight = 0.0, noise_std = 0.01, K = 200, W = 24, H = 23, M = 1, w = 1, h = 1, Ni = 100, Cin = 3, border_mode = "VALID", pool_t_mode = "mean_t")
# model.Compile()
# model.Optimize()
# print model.layer.output_lab.get_shape().as_list()
# b = model.momentum_bn
# print model.survive_thres
# # with tf.Session() as sess:
# # 	feed_dict = {
# # 			model.lr : 0.1,
# # 			model.is_train : 1.0,
# # 			model.momentum_bn	: 0.99,
# # 			model.x_lab : lab_img,
# # 			model.x_unl : lab_img,
# # 			model.x_clean : lab_img,
# # 			model.y_lab : y
# # 			}
# # 	sess.run(tf.initialize_all_variables())
# # 	s = sess.run(model.top_output, feed_dict = feed_dict)
# # 	print s.shape

batch_size =  100
Cin = 1
W = 28
H = 28
seed = 23

model = Model(batch_size, Cin, W, H, seed, is_sup = True)
model.add(noise_weight = 0.0, noise_std = 0.01, K = 1, W = 28, H = 28, M = 1, w = 3, h = 3, Ni = 100, Cin = 1, border_mode = "VALID")
model.add(noise_weight = 0.0, noise_std = 0.01, K = 1, W = 64, H = 64, M = 1, w = 3, h = 3, Ni = 100, Cin = 1, border_mode = "VALID")
model.add(noise_weight = 0.0, noise_std = 0.01, K = 1, W = 64, H = 64, M = 1, w = 3, h = 3, Ni = 100, Cin = 1, border_mode = "VALID")
# print tf.trainable_variables()
# print model.layer.output_lab.get_shape()
model.Compile()
model.Optimize()
epochs = 1
with  tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	summary_op = tf.summary.merge_all()
	writer = tf.summary.FileWriter("logs")
	writer.add_graph(sess.graph)
	batch_no = 0
	for epoch in range(epochs):
		
		# print
		while True:

			try:
				train = mnist.train.next_batch(100)
				# print train[1]
				# break
				feed_dict = {
							model.x_lab : reshape_data(train[0]),
							model.y_lab : train[1],
							model.lr : 0.01,
							model.momentum_bn : 0.99,
							model.is_train : 1
							# self.model.x_unl : self.reshape_data(train[0]),
							# self.model.x_clean : self.reshape_data(train[0])
				}
				print "here"
				_, train_loss, err, summ, probs = sess.run([model.train_ops, model.cost, model.classification_error, summary_op, model.softmax_layer_nonlin.gammas_lab], feed_dict = feed_dict)			
				print "gogo"
				writer.add_summary(summ, batch_no*100)
				# err = sess.run(self.model.cost, feed_dict={self.model.x_lab: self.reshape_data(self.mnist.test.images), self.model.y_lab : self.mnist.test.labels})
															# self.model.x_unl : None, self.model.x_clean : None											
# self.																	})
				print train_loss, err,batch_no*100
				print "train error after training %s batches is %s".format(batch_no, train_loss)
				batch_no += 1

			except Exception as e:
				raise e	
	
print model.layer.output_lab.get_shape().as_list()
