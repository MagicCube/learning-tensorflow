import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def genTrainDataSet(count, weight, bias):
    x = np.linspace(-1, 1, count)
    noise = np.random.rand(count) * 0.3
    y = x * weight + bias + noise
    return (x.astype(np.float32), y.astype(np.float32))


x = tf.placeholder(tf.float32, shape=[None, ])
y = tf.placeholder(tf.float32, shape=[None, ])
weight = tf.Variable(tf.random_uniform([1, ], -1., 1.))
bias = tf.Variable(0.1)
pred = x * weight + bias
loss = tf.reduce_mean((pred - y) ** 2)
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


xs, ys = genTrainDataSet(100, 0.6, 2)
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
