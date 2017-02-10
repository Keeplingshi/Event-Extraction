'''
Created on 2017年2月10日
双向lstm识别触发词方法封装
@author: chenbin
'''

import pickle
import tensorflow as tf


ace_data_test_file = open('./corpus_deal/ace_data/ace_data_test.pkl', 'rb')
ace_data_test = pickle.load(ace_data_test_file)

ace_data_test_labels_file = open(
    './corpus_deal/ace_data/ace_data_test_labels.pkl', 'rb')
ace_data_test_labels = pickle.load(ace_data_test_labels_file)

ace_data_test_file.close()
ace_data_test_labels_file.close()

# 神经网络的参数
n_input = 200  # 输入层的n
n_steps = 1  # 28长度
n_hidden = 128  # 隐含层的特征数
n_classes = 34  # 输出的数量，因为是分类问题，这里一共有34个

sess = tf.InteractiveSession()
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

W1 = tf.Variable(tf.random_normal([2 * n_hidden, n_classes]), name="W1")
b1 = tf.Variable(tf.random_normal([n_classes]), name="b1")

saver = tf.train.Saver()
saver.restore(sess, "./ckpt_file/ace_bl.ckpt")


def BiRNN(x, weights, biases):
    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshape to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    try:
        outputs, _, _ = tf.nn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                dtype=tf.float32)
    except Exception:  # Old TensorFlow version only returns outputs not states
        outputs = tf.nn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                          dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights) + biases

pred = BiRNN(x, W1, b1)

# print(sess.run(W1))
print('1----------------------------------------')
print(sess.run(W1))
# Initializing the variables
# init = tf.global_variables_initializer()
# sess.run(init)

# print(sess.run(W1))
# 载入测试集进行测试
length = len(ace_data_test)
test_accuracy = 0.0
p_s = 0  # 识别的个体总数
r_s = 0  # 测试集中存在个个体总数
pr_acc = 0  # 正确识别的个数

for i in range(length):
    test_len = len(ace_data_test[i])
    test_data = ace_data_test[i].reshape((-1, n_steps, n_input))  # 8
    test_label = ace_data_test_labels[i]
    # prediction识别出的结果，y_测试集中的正确结果
    prediction, y_ = sess.run(
        [tf.argmax(pred, 1), tf.argmax(y, 1)], feed_dict={x: test_data, y: test_label})
    for t in range(len(y_)):
        if prediction[t] != 33:
            p_s = p_s + 1

        if y_[t] != 33:
            r_s = r_s + 1
            if y_[t] == prediction[t]:
                pr_acc = pr_acc + 1

print('----------------------------------------------------')
print('共识别出：' + str(p_s))
print('识别正确：' + str(pr_acc))
print('触发词总数：' + str(r_s))
p = pr_acc / p_s
r = pr_acc / r_s
f = 2 * p * r / (p + r)
print('P=' + str(p) + "\tR=" + str(r) + "\tF=" + str(f))

sess.close()
