import unittest
# import tenosrflow as tf
# import theano
import numpy as np
# from .. layer import *

class LayerTest(unittest.TestCase):
	"""docstring for LayerTest"""
	def get_tf_output(self, inp):
		pass
	def get_theano_output(self, inp):
		pass
	def test_layer(self):
		lab_img = np.random.rand(100,9,8,3)
		unlab_img = np.random.rand(100,9,8,3)
		unlab_img_clean = np.random.rand(100,8,8,3)
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

		tf_layer = Layer(lab_img, unlab_img, unlab_img_clean, noise_wt, noise_std, K, M, W, H, w, h, Cin, Ni, momentum_bn, is_train)
		print "gog"
		# theano_layer = CRM(lab_img, unlab_img, unlab_img_clean, noise_wt, noise_std, K, M, W, H, w, h, Cin, Ni, momentum_bn, is_train)

if __name__ == "__main__":
	# unittest.main()
	print np.random.rand(100,8,8,3).shape