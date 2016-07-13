# Linear Regression

import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

def prepare_data():
	x = np.linspace(-1, 1, 20)
	y = [(2 + random.random()) * xi + 3 + random.random() for xi in x]
	return x, y

def model(x, w, b):
	return tf.mul(x, w) + b

def fit(input_x, input_y, learning_rate, Num_epoch):
	# create symbolic variables
	X = tf.placeholder('float')
	Y = tf.placeholder('float')
	
	# create model parameters and a linear model
	w = tf.Variable(np.random.randn(), name = 'weights')
	b = tf.Variable(np.random.randn(), name = 'bias')
	y_pred = model(X, w, b)

	# define cost = ||y_pred - Y||^2
	cost = tf.pow(Y - y_pred, 2)
	cost = tf.reduce_sum(cost)

	# define the optimizer
	train_opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
	
	# optimize
	sess = tf.Session()
	init = tf.initialize_all_variables()
	sess.run(init)

	assert len(input_x) == len(input_y)
	for epoch_i in xrange(Num_epoch):
		for (xi, yi) in zip(input_x, input_y):
			sess.run(train_opt, feed_dict = {X: xi, Y: yi})
		print 'Epoch %d: w = %f, b = %f' %(epoch_i, sess.run(w), sess.run(b))
	return sess.run(w), sess.run(b)

def plot(x, y, w, b):
	plt.plot(x, y, 'ro', label = 'data source')
	plt.plot(x, w * x + b, 'g.-', label = 'fitting curve')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	datax, datay = prepare_data()
	w, b = fit(datax, datay, 0.01, 10)
	plot(datax, datay, w, b)
