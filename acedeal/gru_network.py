'''
Created on 2017年2月10日

@author: chenbin
'''
import pickle
import tensorflow as tf
# from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

# 数据读取，训练集和测试集
ace_data_train_file = open('./corpus_deal/ace_data/ace_data_train.pkl', 'rb')
ace_data_train = pickle.load(ace_data_train_file)

ace_data_train_labels_file = open(
    './corpus_deal/ace_data/ace_data_train_labels.pkl', 'rb')
ace_data_train_labels = pickle.load(ace_data_train_labels_file)

ace_data_test_file = open('./corpus_deal/ace_data/ace_data_test.pkl', 'rb')
ace_data_test = pickle.load(ace_data_test_file)

ace_data_test_labels_file = open(
    './corpus_deal/ace_data/ace_data_test_labels.pkl', 'rb')
ace_data_test_labels = pickle.load(ace_data_test_labels_file)

ace_data_train_file.close()
ace_data_train_labels_file.close()
ace_data_test_file.close()
ace_data_test_labels_file.close()

# call mnist function

learningRate = 0.001
training_iters = 16000
batchSize = 1
displayStep = 10

nInput = 200  # we want the input to take the 28 pixels
nSteps = 1  # every 28
nHidden = 128  # number of neurons for the RNN 64 128 256
nClasses = 34  # this is MNIST so you know

x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])

weights = {
    'out': tf.Variable(tf.random_normal([2 * nHidden, nClasses]))
}

biases = {
    'out': tf.Variable(tf.random_normal([nClasses]))
}


def gru_RNN(x, weights, biases):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, nInput])
    # configuring so you can get it as needed for the 28 pixels
    x = tf.split(0, nSteps, x)

#     gruCell = tf.nn.rnn_cell.GRUCell(nHidden)
    #gruCell = rnn_cell.GRUCell(nHidden)
    #rnnCell = rnn_cell.BasicRNNCell(nHidden)
    #lstmCell = rnn_cell.BasicLSTMCell(nHidden, forget_bias=1.0)
    #outputs, states = rnn.rnn(rnnCell, x, dtype=tf.float32)
    #outputs, states = rnn.rnn(lstmCell, x, dtype=tf.float32)

    # Define gru cells with tensorflow
    # Forward direction cell
    gru_fw_cell = tf.nn.rnn_cell.GRUCell(nHidden)
    # Backward direction cell
    gru_bw_cell = tf.nn.rnn_cell.GRUCell(nHidden)

    # Get gru cell output
    try:
        outputs, _, _ = tf.nn.bidirectional_rnn(gru_fw_cell, gru_bw_cell, x,
                                                dtype=tf.float32)
    except Exception:  # Old TensorFlow version only returns outputs not states
        outputs = tf.nn.bidirectional_rnn(gru_fw_cell, gru_bw_cell, x,
                                          dtype=tf.float32)

    #outputs, states = tf.nn.rnn(gruCell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = gru_RNN(x, weights, biases)

# optimization
# create the cost, optimization, evaluation, and accuracy
# for the cost softmax_cross_entropy_with_logits seems really good
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

correctPred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

init = tf.global_variables_initializer()
train_accuracy_array, train_loss_array = [], []
with tf.Session() as sess:
    sess.run(init)
    k = 0

    while k < training_iters:
        step = k % 1600

        batch_xs = ace_data_train[step]
        batch_ys = ace_data_train_labels[step]
        batch_size = len(batch_xs)
        batch_xs = batch_xs.reshape([batch_size, nSteps, nInput])

        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

        if step % 100 == 0:
            #             acc = sess.run(accuracy, feed_dict={x: batchX, y: batchY})
            #             loss = sess.run(cost, feed_dict={x: batchX, y: batchY})
            #             train_accuracy_array.append(acc)
            #             train_loss_array.append(loss)
            print("Iter " + str(k))
        k += 1
    print('Optimization finished')

    # 载入测试集进行测试
    length = len(ace_data_test)
    test_accuracy = 0.0
    p_s = 0  # 识别的个体总数
    r_s = 0  # 测试集中存在个个体总数
    pr_acc = 0  # 正确识别的个数
#     s = 0
#     acc = 0
    for i in range(length):
        test_len = len(ace_data_test[i])
        test_data = ace_data_test[i].reshape(
            (-1, nSteps, nInput))  # 8
        test_label = ace_data_test_labels[i]
        # prediction识别出的结果，y_测试集中的正确结果
        prediction, y_ = sess.run(
            [tf.argmax(pred, 1), tf.argmax(y, 1)], feed_dict={x: test_data, y: test_label})
        for t in range(len(y_)):
            if prediction[t] != 33:
                p_s = p_s + 1

            if y_[t] != 33:
                r_s = r_s + 1
#                 logging.info(
#                     'actual:' + str(y_[t]) + '\t predict:' + str(prediction[t]))
#                 #print(str(y_[t]) + '\t' + str(prediction[t]))
#                 s = s + 1
                if y_[t] == prediction[t]:
                    pr_acc = pr_acc + 1


#     logging.info('----------------------------------------------------')
#     logging.info(str(acc) + '------------' + str(s))
#     logging.info(acc / s)
    print('----------------------------------------------------')
    print(str(pr_acc) + '------------' + str(r_s))
    p = pr_acc / p_s
    r = pr_acc / r_s
    f = 2 * p * r / (p + r)
    print('P=' + str(p) + "\tR=" + str(r) + "\tF=" + str(f))


# 299------------579
# P=0.5567970204841713    R=0.5164075993091537    F=0.5358422939068099


# 301------------579
# P=0.5289982425307557    R=0.5198618307426598    F=0.524390243902439
