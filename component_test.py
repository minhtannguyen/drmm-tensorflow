import theano
import theano.tensor as T
import tensorflow as tf
import numpy as np
from drmmt.utils import *
from drmm.nn_functions_latest import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def compare(a, b):
	print "ff", a.shape
	a = np.array(a).reshape((-1))
	print "hh"
	b = np.array(b).reshape((-1))
	print "hi"
	for i in range(a.shape[0]):
		print "hi", a[i], b[i]
		if (abs(a[i])-abs(b[i]) > 1e-4):
			print a[i],b[i]
			print i
			# return False
	return True


# def compare(a, b):
# 	print "ff", a.shape
# 	a = np.array(a)
# 	print "hh"
# 	b = np.array(b)
# 	print "hi"
# 	f = b.shape
# 	print f,f[0],f[1],f[2],f[3],a.shape
# 	for i in range(a.shape[0]):
# 		for j in range(a.shape[1]):
# 			for k in xrange(a.shape[2]):
# 				for m in range(a.shape[3]):
# 					print "hi",i,j,k,m, a[i,j,k][m],b[i,j,k][m]
# 					if (a[i,j,k][m]-b[i,j,k][m] > 1):
# 						print a[i,j,k][m],b[i,j,k][m]
# 						# print i
# 						return False
# 	return True




lab_img_np = np.random.rand(100, 8, 9, 8)
unlab_img_np = np.random.rand(100, 3, 9, 8)
unlab_img_clean_np = np.random.rand(100, 3, 9, 8)
lambdas = np.random.rand(8,27,1)
noise_wt = 0.0
noise_std = 0.01
K = 8
W = 8
H = 9
M = 1
w = 3
h = 3
Cin = 3
Ni = 100
momentum_bn = 0.99
is_train = 1
y = np.random.rand(100,8)


a = BatchNormalization1(insize=K, mode=1, momentum=momentum_bn, is_train=is_train,
											epsilon=1e-10,
											gamma_val_init=None, beta_val_init=None,
											mean_init=None, var_init=None)
s = a.get_result(tf.convert_to_tensor(lab_img_np,dtype = tf.float32), [100,8,9,8])


b = BatchNormalization(insize=K, mode=1, momentum=momentum_bn, is_train=is_train,
											epsilon=1e-10,
											gamma_val_init=None, beta_val_init=None,
											mean_init=None, var_init=None)
s1 = b.get_result(theano.shared(np.asarray(lab_img_np, dtype=theano.config.floatX)),np.asarray(lab_img_np, dtype=theano.config.floatX).shape)


# fn = theano.function([],b.mean)
# th_o = fn()
# sess= tf.Session()
# sess.run(tf.initialize_all_variables())
# tf_o = sess.run(a.mean, feed_dict = {})



fn = theano.function([],s1)
th_o = fn()
sess= tf.Session()
sess.run(tf.initialize_all_variables())
tf_o = sess.run(s, feed_dict = {})

print compare(th_o, tf_o)