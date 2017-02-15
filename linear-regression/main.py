import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from input_data import gen_train_dataSet


x = tf.placeholder(tf.float32, shape=[None, ])
y = tf.placeholder(tf.float32, shape=[None, ])
weight = tf.Variable(tf.random_uniform([1, ], -1., 1.))
bias = tf.Variable(0.1)
pred = x * weight + bias
loss = tf.reduce_mean((pred - y) ** 2)
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


xs, ys = gen_train_dataSet(100, 0.6, 2)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in xrange(100):
        sess.run(train, feed_dict={x: xs, y: ys})

    line_x = [-2, 2]
    line_y = sess.run(pred, feed_dict={x: line_x})
    plt.scatter(xs, ys)
    plt.plot(line_x, line_y, c='red')
    plt.show()
    print(sess.run(loss, feed_dict={x: xs, y: ys}))
