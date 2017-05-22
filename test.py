import theano
import theano.tensor as T
import tensorflow as tf
import numpy as np


def change_shape(self, vec, input_shape):
    return T.repeat(vec, input_shape[2] * input_shape[3]).reshape(
        (input_shape[1], input_shape[2], input_shape[3]))

v = 2.0*np.ones(10,3,8,8)
# print v
a = 2.0*tf.ones([10,3,8,8])
# b = T.shared(value = v, borrow = True)
c = T.scalar("fdfd")
d = T.mean(v, axis = (0,2,3))
e = T.var(v, axis = (0,2,3))
f = theano.function([c], d)
print f(1)
now_mean, now_var = tf.nn.moments(a, axes=[0, 2, 3])
with tf.Session() as sess:
	print sess.run(d, feed_dict = {})
