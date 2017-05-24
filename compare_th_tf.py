import theano
import tensorflow as tf
import numpy as np
# from drmmt.layer import *
from drmm.CRM_no_factor_latest import *
from drmmt.layer_debug import *
from drmmt.utils import *


def compare(a, b):
	print "ff", a.shape
	a = np.array(a).reshape((-1))
	print "hh"
	b = np.array(b).reshape((-1))
	print "hi"
	for i in range(a.shape[0]):
		# print "hi"
		if (a[i]-b[i] > 1e-3):
			print a[i],b[i]
			print i
			return False
	return True



# def compare(a, b):
# 	print "ff", a.shape
# 	a = np.array(a)
# 	print "hh"
# 	b = np.array(b).reshape(a.shape)
# 	print "hi"
# 	f = b.shape
# 	print f,f[0],f[1],f[2],f[3]
# 	for i in range(a.shape[0]):
# 		for j in range(a.shape[1]):
# 			for k in xrange(a.shape[2]):
# 				for m in range(a.shape[3]):
# 					print "hi",i,j,k,m, a[i,j,k][m],b[i,j,k][m]
# 					if (a[i,j,k][m]-b[i,j,k][m] > 1e-3):
# 						print a[i,j,k][m],b[i,j,k][m]
# 						# print i
# 						return False
# 	return True


# Initial parameters
lab_img_np = np.random.rand(100, 3, 9, 8)
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


# checked = [th_layer.latents_shape, th_layer.D, th_layer.pi_t,th_layer.pi_a,th_layer.prun_mat,th_layer.prun_synap_mat,th_layer.lambdas, th_layer.sigma_dn]
#Theano outputs
th_layer  = CRM(
	lab_img_np, unlab_img_np, unlab_img_clean_np, noise_wt, noise_std, K, M, W, H, w, h, Cin, Ni, momentum_bn, is_train,
	lambdas_val_init = lambdas
)
th_layer._E_step_Bottom_Up()
# init_vars = [th_layer.pi_t, th_layer.pi_a, th_layer.pi_a_old, th_layer.pi_ta, th_layer.prun_mat, th_layer.prun_synap_mat,th_layer.pi_synap,th_layer.pi_synap_old,th_layer.lambdas,th_layer.sigma_dn ,th_layer.b_dn,th_layer.alpha_dn]
init_vars = [th_layer.latents_before_BN_lab, th_layer.latents_lab, th_layer.max_over_a_mask_lab,th_layer.max_over_t_mask_lab, th_layer.latents_masked_lab, th_layer.masked_mat_lab, th_layer.output_lab, th_layer.latents_demeaned_lab, th_layer.latents_demeaned_squared_lab]
# init_vars = [th_layer.c1]
# to_show_vars =[th_layer.latents_before_BN_lab, th_layer.latents_lab, th_layer.max_over_a_mask_lab,th_layer.max_over_t_mask_lab, th_layer.latents_masked_lab, th_layer.masked_mat_lab, th_layer.output_lab, th_layer.mask_input_lab, th_layer.scale_s_lab,th_layer.latents_demeaned_lab, th_layer.latents_demeaned_squared_lab]
outs = []
# for i in range(2):
# 	outs.append(to_show_vars[i])
# for i in range(len(to_show_vars)):
for i in range(len(init_vars)):
	# break
	fn = theano.function([],  init_vars[i])
	outs.append(fn())
#Tensorflow outputs
lab_img = tf.placeholder(tf.float32, [100,3,9,8])
unlab_img = tf.placeholder(tf.float32, [100,3,9,8])
unlab_img_clean = tf.placeholder(tf.float32, [100,3,9,8])
tf_layer = Layer(lab_img, unlab_img, unlab_img_clean, noise_wt, noise_std, K, M, W, H, w, h, Cin, Ni, momentum_bn, is_train, lambdas_t_val_init = lambdas)
tf_layer.EBottomUp()
# print tf_layer.output_lab.get_shape(),"hore"

# RegressionInSoftmax = HiddenLayerInSoftmax(input_lab=tf.reshape(tf_layer.output_lab,[Ni,-1]),input_unl=tf.reshape(tf_layer.output,[Ni,-1]),input_clean=tf.reshape(tf_layer.output_clean,[Ni,-1]),n_in=72, n_out=8,W_init=None, b_init=None)
# softmax_input_lab = RegressionInSoftmax.output_lab
# softmax_input = RegressionInSoftmax.output
# softmax_input_clean = RegressionInSoftmax.output_clean
# softmax_layer_nonlin = SoftmaxNonlinearity(input_lab=softmax_input_lab, input_unl=softmax_input,input_clean=softmax_input_clean)
# ss= softmax_layer_nonlin.y_pred_lab
# init_vars = [tf_layer.pi_t, tf_layer.pi_a, tf_layer.pi_a_old, tf_layer.pi_ta, tf_layer.prun_mat, tf_layer.prun_synap_mat,tf_layer.pi_synap,tf_layer.pi_synap_old,tf_layer.lambdas_t,tf_layer.sigma_dn ,tf_layer.b_dn,tf_layer.alpha_dn]
init_vars = [tf_layer.latents_before_BN_lab, tf_layer.latents_lab, tf_layer.max_over_a_mask_lab,tf_layer.max_over_t_mask_lab, tf_layer.latents_masked_lab, tf_layer.masked_mat_lab, tf_layer.output_lab,tf_layer.latents_demeaned_lab, tf_layer.latents_demeaned_squared_lab]
# init_vars = [tf_layer.c1]
# to_show_vars =[tf_layer.latents_before_BN_lab, tf_layer.latents_lab, tf_layer.max_over_a_mask_lab,tf_layer.max_over_t_mask_lab, tf_layer.latents_masked_lab, tf_layer.masked_mat_lab, tf_layer.output_lab, tf_layer.mask_input_lab, tf_layer.scale_s_lab,tf_layer.latents_demeaned_lab, tf_layer.latents_demeaned_squared_lab]
# to_show_vars1 = []
# output = to_show_vars[:2]
output = []
# for i in 
# tf_layer.ETopDown()
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	# break
	output += sess.run(init_vars, feed_dict = {lab_img : lab_img_np, unlab_img : unlab_img_np, unlab_img_clean : unlab_img_clean_np})
	# print s


#compare
for i in range(len(output)):
	print "var_no : ", i
	# print outs[i]," ,",output[i]
	try:
		# if (i==6):
		# 	print outs[i],"fdfdfd  \n"
		# 	print output[i]
		assert(np.array_equal(outs[i],output[i]))	

	except Exception as e:
		try:
			# print "gog"
			assert(compare(output[i], outs[i]))	
		except Exception as f:
			print f
			print outs[i].shape, output[i].shape	
	# break
		

	