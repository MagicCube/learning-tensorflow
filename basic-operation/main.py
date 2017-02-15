import tensorflow as tf

delta = tf.placeholder(tf.float32)
result = tf.Variable(0.)
accumulation = tf.assign_add(result, delta)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in xrange(10):
        sess.run(accumulation, feed_dict={delta: 1.})
        print(sess.run(result))
