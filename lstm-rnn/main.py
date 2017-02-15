import tensorflow as tf

from input_data import InputData

LEARNING_RATE = 0.001
TRAIN_ITERATION_COUNT = 50
TRAIN_BATCH_SIZE = 100                      # How many sequences are there in a single training batch
MAX_SEQ_LEN = 20                            # Length of the longest words

INPUT_RANK = len(InputData.ALPHABETS)       # Rank of the input, here we use a 10-dim one-hot data as input,
OUTPUT_RANK = INPUT_RANK                    # and the same form as output
HIDDEN_UNIT_COUNT = OUTPUT_RANK * 4         # How many hidden units are there inside a RNN cell


x = tf.placeholder(tf.float32, [None, MAX_SEQ_LEN, INPUT_RANK], name='x_input')
y = tf.placeholder(tf.float32, [None, OUTPUT_RANK], name='y_input')
seq_len_of_x = tf.placeholder(tf.float32, [None, ], name='sequence_length_of_input')
weights = {
    'in': tf.Variable(tf.random_normal([INPUT_RANK, HIDDEN_UNIT_COUNT], name='weights_in')),
    'out': tf.Variable(tf.random_normal([HIDDEN_UNIT_COUNT, OUTPUT_RANK]), name='weights_out')
}
biases = {
    'in': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[HIDDEN_UNIT_COUNT, ]), name='biases_in'),
    'out': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[OUTPUT_RANK, ]), name='biases_out'),
    'forget': 1.0
}


def rnn(inputs, seq_len_of_inputs, weights, biases):
    # inputs[?, MAX_SEQ_LEN, INPUT_RANK] => [? * MAX_SEQ_LEN, INPUT_RANK]
    inputs = tf.reshape(inputs, [-1, INPUT_RANK])
    # cell_inputs[? * MAX_SEQ_LEN, CELL_SIZE]
    cell_inputs = tf.tanh(tf.matmul(inputs, weights['in']) + biases['in'])
    # cell_inputs[?, MAX_SEQ_LEN, CELL_SIZE]
    cell_inputs = tf.reshape(cell_inputs, [-1, MAX_SEQ_LEN, HIDDEN_UNIT_COUNT])
    cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_UNIT_COUNT, forget_bias=biases['forget'])
    cell_outputs, cell_states = tf.nn.dynamic_rnn(cell, cell_inputs, sequence_length=seq_len_of_inputs, dtype=tf.float32)
    cell_c_states, cell_m_states = cell_states
    results = tf.matmul(cell_m_states, weights['out']) + biases['out']
    return results

pred = rnn(x, seq_len_of_x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))




with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in xrange(TRAIN_ITERATION_COUNT):
        train_x, train_seq_len_of_x, train_y = InputData.next_batch(TRAIN_BATCH_SIZE, min_len=1, max_len=20)
        sess.run(train, feed_dict={x: train_x, y: train_y, seq_len_of_x: train_seq_len_of_x})
        if (i + 1) % 10 == 0:
            print('%.2f%%, accuracy = %.2f%%' %
            ((i + 1) * 1.0 / TRAIN_ITERATION_COUNT * 100,
            sess.run(accuracy, feed_dict={x: train_x, seq_len_of_x: train_seq_len_of_x, y: train_y}) * 100))



    test_input = 'h'
    test_x = InputData.lettersToBinVectorList(test_input)
    test_x = InputData.pad(test_x)
    test_seq_len_of_x = len(test_input)
    results = sess.run(tf.nn.softmax(pred), feed_dict={x: [test_x], seq_len_of_x: [test_seq_len_of_x]})
    print('The input is "%s"' % test_input)
    print('The next letter will be "%s"' % InputData.ALPHABETS[results.argmax()])
    print(results[0])

