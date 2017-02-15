import tensorflow as tf

from input_data import InputData


CLASS_COUNT = FEATURE_COUNT = len(InputData.ALPHABETS)
HIDDEN_UNIT_COUNT = CLASS_COUNT * 2
TRAIN_BATCH_SIZE = 10
LEARNING_RATE = 0.001


x = tf.placeholder(tf.float32, [None, FEATURE_COUNT])
y = tf.placeholder(tf.float32, [None, CLASS_COUNT])

weights = {
    'in': tf.Variable(tf.random_normal([FEATURE_COUNT, HIDDEN_UNIT_COUNT])),
    'out': tf.Variable(tf.random_normal([HIDDEN_UNIT_COUNT, CLASS_COUNT]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[HIDDEN_UNIT_COUNT, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[CLASS_COUNT, ])),
    'forget': 1.0
}


def rnn(X, weights, biases):
    # x_in[?, HIDDEN_UNIT_COUNT]
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in[?, HIDDEN_UNIT_COUNT]
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_UNIT_COUNT)
    lstm_initial_state = lstm_cell.zero_state(TRAIN_BATCH_SIZE, tf.float32)
    outputs, (c_state, m_state) = tf.nn.rnn(lstm_cell, X_in, initial_state=lstm_initial_state)
    #results[?, CLASS_COUNT]
    results = tf.matmul(m_state, weights['out']) + biases['out']
    return results


pred = rnn(x, weights, biases)
lost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(lost)

with tf.Session() as sess:
    InputData.ALPHABETS = ALPHABETS
    sess.run(tf.global_variables_initializer())
    xs, ys = InputData.next_batch(TRAIN_BATCH_SIZE)
    sess.run(train, feed_dict={x: xs, y: ys})

