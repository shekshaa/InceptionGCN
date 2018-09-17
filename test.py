import tensorflow as tf
import numpy as np
from layers import *

# name = 'weight'
# shape = [3, 4]
# init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
# initial_real = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
# initial_img = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
# c = tf.Variable(tf.complex(initial_real, initial_img, name=name))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# r = tf.reduce_sum(tf.real(c))
# opt = optimizer.minimize(r)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     sess.run(opt)
#

import hickle as hkl

# placeholders = {
#     'laplacian': tf.sparse_placeholder(tf.float32, shape=[3, 3]),
#     'adj': tf.sparse_placeholder(tf.float32, shape=[3, 3]),
#     'degree': tf.sparse_placeholder(tf.float32, shape=[3, 3]),
#     'features': tf.placeholder(tf.float32),
#     # 'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
#     # 'labels_mask': tf.placeholder(tf.int32),
#     'dropout': tf.placeholder_with_default(0., shape=()),
#     'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
# }
#
# layer = GraphConvolutionCayley(input_dim=3, output_dim=1, locality_size=3, num_nodes=500, placeholders=placeholders, approx_step=5,
#                                zoom=1., sparse_inputs=False)
#
# layer(placeholders['features'])

