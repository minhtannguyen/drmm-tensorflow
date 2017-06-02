import numpy as np
import tensorflow as tf
import sys
import os 

sys.path.insert(0, '/home/rgoel/drmm/theano_implementation/drmmt')
sys.path.insert(0, '/home/rgoel/drmm/theano_implementation/drmm')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from model import *
from the_model import *

save_dir = "save/testing"

print "hoho"
class ModelTest(object):
	"""docstring for ModelTest"""

	def compare(self, a, b):
		a_sh = a.shape
		b_sh = b.shape
		# print len(a_sh)
		if len(a_sh) == 1:
			for i in range(a.shape[0]):
				print a[i]-b[i]
				if (abs(a[i]-b[i]) > 1e-4):
					# print a[i],b[i]
					# print i
					return False
		elif len(a_sh) == 2:
				for i in range(a.shape[0]):
					for j in range(a.shape[1]):
						print a[i,j]-b[i,j]
						if (abs(a[i,j]-b[i,j]) > 1e-4):
							
							return False
		elif len(a_sh) == 3:
				for i in range(a.shape[0]):
					for j in range(a.shape[1]):
						for k in xrange(a.shape[2]):
							# for m in range(a.shape[3]):
							# print "hi",i,j,k, a[i,j,k],b[i,j,k], a[i,j,k]==b[i,j,k]
							print a[i,j,k]-b[i,j,k] 
							if (abs(a[i,j,k]-b[i,j,k]) > 1e-4):
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
							print (a[i,j,k][m])-(b[i,j,k][m])
							if (abs((a[i,j,k][m])-(b[i,j,k][m])) > 1e-4):
								# print a[i,j,k][m],b[i,j,k][m]
								return False
			
		return True


	def check(self, tf, th, no):
		for i in range(no):
			print i
			try:
				print "hi"
				assert(np.array_equal(tf[i],th[i]))
			except Exception as e:
				try:
					print "yo"
					assert(self.compare(tf[i], th[i]))
					print "yo"
					# break
				except Exception as e:
					assert(np.array_equal(tf[i],th[i]))
					print "break ooccured at ",i
					break
			if i ==2 or i==3:
				print th[i],tf[i]



	def tf_model(self, lab_img, unlab_img, unlab_img_clean, y_labs, batch_size =  100, Cin = 1, W = 28, H = 28, seed = 23, epochs = 1):
		tf_model = Model(batch_size, Cin, W, H, seed)
		tf_model.add(noise_weight = 0.0, noise_std = 0.01, K = 32, W = 28, H = 28, M = 1, w = 3, h = 3, Ni = 100, Cin = 1, border_mode = "VALID")
		tf_model.add(noise_weight = 0.0, noise_std = 0.01, K = 64, W = 64, H = 64, M = 1, w = 3, h = 3, Ni = 100, Cin = 1, border_mode = "VALID")
		tf_model.add(noise_weight = 0.0, noise_std = 0.01, K = 100, W = 64, H = 64, M = 1, w = 3, h = 3, Ni = 100, Cin = 1, border_mode = "VALID")
		tf_model.Compile()
		tf_model.Optimize()
		model_vars = [ tf_model.softmax_layer_nonlin.y_pred, tf_model.softmax_layer_nonlin.y_pred_lab, tf_model.classification_error, tf_model.cost]
		for i in range(tf_model.N_layers):
			model_vars.append(tf_model.layers[i].data_reconstructed_lab)
			model_vars.append(tf_model.layers[i].reconstruction_error_lab)
			model_vars.append(tf_model.layers[i].reconstruction_error)

		with tf.Session() as sess:
			sess.run(tf.initialize_all_variables())
			feed_dict = {
						tf_model.x_lab : lab_img,
						tf_model.x_unl : unlab_img,
						tf_model.x_clean : unlab_img_clean,
						tf_model.y_lab : y_labs,
						tf_model.lr : 0.00001,
						tf_model.momentum_bn : 0.99,
						tf_model.is_train : 1
					
			}
			# print "here"
			outs = sess.run(model_vars, feed_dict = feed_dict)			
			return outs
				


	def th_model(self, lab_img, unlab_img, unlab_img_clean, y_labs,batch_size =  100, Cin = 1, W = 28, H = 28, seed = 23):
		th_model = ModelT(batch_size, Cin, W, H, seed)
		
		model_vars = [ th_model.softmax_layer_nonlin.y_pred, th_model.softmax_layer_nonlin.y_pred_lab, th_model.classification_error, th_model.cost]
		# for i in range(th_model.N_layer):
		# 	print i
		# 	model_vars.append(th_model.layers[i].data_reconstructed_lab)
		# 	model_vars.append(th_model.layers[i].reconstruction_error_lab)
		# 	model_vars.append(th_model.layers[i].reconstruction_error)
		train_ops = theano.function(inputs =  [th_model.x_lab, th_model.x_unl, th_model.y_lab],outputs = model_vars, on_unused_input='warn')
		outs = train_ops( lab_img, unlab_img, y_labs)
		return outs

	def test_model(self, batch_size =  100, Cin = 1, W = 28, H = 28, seed = 23):
		lab_img = np.random.rand(100, 1, 28, 28)
		unlab_img = np.random.rand(100, 1, 28, 28)
		unlab_img_clean = np.random.rand(100, 1, 28, 28)
		y_labs = np.random.randint(0,9,100, dtype = "int32")


		tf_outs = self.tf_model(lab_img, unlab_img, unlab_img_clean, y_labs, batch_size , Cin , W , H, seed)
		th_outs = self.th_model(lab_img, unlab_img, unlab_img_clean, y_labs, batch_size , Cin , W , H, seed)
		self.check(tf_outs, th_outs, 4)
		print "yeah"

ModelTest().test_model()	