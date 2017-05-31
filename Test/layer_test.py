import unittest
import tensorflow as tf
import theano

import numpy as np


import sys
import os 

sys.path.insert(0, '/home/rgoel/drmm/theano_implementation/drmmt')
sys.path.insert(0, '/home/rgoel/drmm/theano_implementation/drmm')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from layer import *
from CRM_no_factor_latest import *


# class LayerTest(unittest.TestCase):
class LayerTest():
	"""docstring for LayerTest"""

	def compare(self, a, b):
		a_sh = a.shape
		b_sh = b.shape
		# print len(a_sh)
		if len(a_sh) == 1:
			for i in range(a.shape[0]):
				if (a[i]-b[i] > 1e-4):
					# print a[i],b[i]
					# print i
					return False
		elif len(a_sh) == 2:
				for i in range(a.shape[0]):
					for j in range(a.shape[1]):
						if (a[i,j]-b[i,j] > 1e-4):
							
							return False
		elif len(a_sh) == 3:
				for i in range(a.shape[0]):
					for j in range(a.shape[1]):
						for k in xrange(a.shape[2]):
							# for m in range(a.shape[3]):
							# print "hi",i,j,k, a[i,j,k],b[i,j,k], a[i,j,k]==b[i,j,k]
							# print a[i,j,k]-b[i,j,k] > 1e-4
							if (a[i,j,k]-b[i,j,k] > 1e-4):
								# print a[i,j,k],b[i,j,k]
								# print i
								return False

		elif len(a_sh) == 4:
			for i in range(a.shape[0]):
				for j in range(a.shape[1]):
					for k in xrange(a.shape[2]):
						for m in range(a.shape[3]):
							# print "hi",i,j,k,m, a[i,j,k][m],b[i,j,k][m]
							# print a[i,j,k][m] == b[i,j,k][m]
							# print (a[i,j,k][m])-(b[i,j,k][m]) > 1e-4
							if ((a[i,j,k][m])-(b[i,j,k][m]) > 1e-4):
								# print a[i,j,k][m],b[i,j,k][m]
								return False
			
		return True

	def check(self, tf, th):
		for i in range(len(tf)):
			print i
			try:
				# print "hi"
				assert(np.array_equal(tf[i],th[i]))
			except Exception as e:
				try:
					# print "yo"
					assert(self.compare(tf[i], th[i]))
					# print "yo"
					# break
				except Exception as e:
					assert(np.array_equal(tf[i],th[i]))
					print "break ooccured at ",i
					break


	def get_tf_output(self, lab_img_np, unlab_img_np, unlab_img_clean_np, noise_wt, noise_std, 
							K, M, W, H, w, h, Cin, Ni, momentum_bn, is_train, lambdas_t):

		lab_img = tf.placeholder(tf.float32, [100,3,9,8])
		unlab_img = tf.placeholder(tf.float32, [100,3,9,8])
		unlab_img_clean = tf.placeholder(tf.float32, [100,3,9,8])
		tf_layer = Layer(lab_img, unlab_img, unlab_img_clean, noise_wt, noise_std, K, M, W, H, w, h, Cin, Ni, momentum_bn, is_train, lambdas_t_val_init = lambdas_t)
		tf_layer.EBottomUp()
	
		inits = [tf_layer.pi_t, tf_layer.pi_a, tf_layer.pi_a_old, tf_layer.pi_ta, tf_layer.prun_mat, tf_layer.prun_synap_mat,tf_layer.pi_synap,tf_layer.pi_synap_old,tf_layer.lambdas_t,tf_layer.sigma_dn ,tf_layer.b_dn,tf_layer.alpha_dn]
		to_run_lab = [tf_layer.betas, tf_layer.latents_before_BN_lab, tf_layer.latents_lab, tf_layer.max_over_a_mask_lab, tf_layer.max_over_t_mask_lab, tf_layer.latents_masked_lab, tf_layer.masked_mat_lab, tf_layer.output_lab, tf_layer.mask_input_lab, tf_layer.scale_s_lab, tf_layer.latents_demeaned_lab, tf_layer.latents_demeaned_squared_lab]
		to_run_unl = [tf_layer.latents_before_BN, tf_layer.latents, tf_layer.max_over_a_mask, tf_layer.max_over_t_mask, tf_layer.latents_masked, tf_layer.masked_mat, tf_layer.output, tf_layer.mask_input, tf_layer.scale_s, tf_layer.latents_demeaned, tf_layer.latents_demeaned_squared]
		others = [tf_layer.pi_t_minibatch, tf_layer.pi_a_minibatch, tf_layer.pi_ta_minibatch, tf_layer.pi_t_new, tf_layer.pi_a_new, tf_layer.pi_ta_new, tf_layer.pi_synap_minibatch]
	
		to_run = inits + to_run_lab + to_run_unl + others
		
		sess = tf.Session()	
		sess.run(tf.global_variables_initializer())
		outs = sess.run(to_run, feed_dict = {lab_img : lab_img_np, unlab_img : unlab_img_np, unlab_img_clean : unlab_img_clean_np})
	
		return outs

	def get_th_output(self, lab_img, unlab_img, unlab_img_clean, noise_wt, noise_std, 
							K, M, W, H, w, h, Cin, Ni, momentum_bn, is_train, lambdas_t):

		th_layer = CRM(lab_img, unlab_img, unlab_img_clean, noise_wt, noise_std, K, M, W, H, w, h, Cin, Ni, momentum_bn, is_train, lambdas_val_init = lambdas_t)
		th_layer._E_step_Bottom_Up()
	
		inits = [th_layer.pi_t, th_layer.pi_a, th_layer.pi_a_old, th_layer.pi_ta, th_layer.prun_mat, th_layer.prun_synap_mat,th_layer.pi_synap,th_layer.pi_synap_old,th_layer.lambdas,th_layer.sigma_dn ,th_layer.b_dn,th_layer.alpha_dn]
		to_run_lab = [th_layer.betas, th_layer.latents_before_BN_lab, th_layer.latents_lab, th_layer.max_over_a_mask_lab, th_layer.max_over_t_mask_lab, th_layer.latents_masked_lab, th_layer.masked_mat_lab, th_layer.output_lab, th_layer.mask_input_lab, th_layer.scale_s_lab, th_layer.latents_demeaned_lab, th_layer.latents_demeaned_squared_lab]
		to_run_unl = [th_layer.latents_before_BN, th_layer.latents, th_layer.max_over_a_mask, th_layer.max_over_t_mask, th_layer.latents_masked, th_layer.masked_mat, th_layer.output, th_layer.mask_input, th_layer.scale_s, th_layer.latents_demeaned, th_layer.latents_demeaned_squared]
		others = [th_layer.pi_t_minibatch, th_layer.pi_a_minibatch, th_layer.pi_ta_minibatch, th_layer.pi_t_new, th_layer.pi_a_new, th_layer.pi_ta_new, th_layer.pi_synap_minibatch]
	
		to_run = inits + to_run_lab + to_run_unl + others
		
		sess = theano.function([], to_run)
		outs = sess()
	
		return outs
		
	def test_layer(self):
		lab_img = np.random.rand(100, 3, 9, 8)
		unlab_img = np.random.rand(100, 3, 9, 8)
		unlab_img_clean = np.random.rand(100, 3, 9, 8)
		lambdas_t = np.random.rand(8,27,1)
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

		tf_outs = self.get_tf_output(lab_img, unlab_img, unlab_img_clean, noise_wt, noise_std, 
							K, M, W, H, w, h, Cin, Ni, momentum_bn, is_train, lambdas_t)	

		print "got tensorflow outputs"
		
		th_outs = self.get_th_output(lab_img, unlab_img, unlab_img_clean, noise_wt, noise_std, 
							K, M, W, H, w, h, Cin, Ni, momentum_bn, is_train, lambdas_t)

		print "Done with running the theano and tensorflow code...\nNow checking whether they match"
		self.check(tf_outs, th_outs)
		# theano_layer = CRM(lab_img, unlab_img, unlab_img_clean, noise_wt, noise_std, K, M, W, H, w, h, Cin, Ni, momentum_bn, is_train)

if __name__ == "__main__":
	# unittest.main()
	print np.random.rand(100,8,8,3).shape
	LayerTest().test_layer()