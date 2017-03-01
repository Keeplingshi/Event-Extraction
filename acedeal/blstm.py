'''
Created on 2017年2月10日

@author: chenbin
'''
import pickle
import tensorflow as tf
# from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

# 数据读取，训练集和测试集
ace_data_train_file = open('./ace_data_process/ace_eng_data1/ace_data_train.pkl', 'rb')
ace_data_train = pickle.load(ace_data_train_file)

ace_data_train_labels_file = open('./ace_data_process/ace_eng_data1/ace_data_train_labels.pkl', 'rb')
ace_data_train_labels = pickle.load(ace_data_train_labels_file)

ace_data_test_file = open('./ace_data_process/ace_eng_data1/ace_data_test.pkl', 'rb')
ace_data_test = pickle.load(ace_data_test_file)

ace_data_test_labels_file = open('./ace_data_process/ace_eng_data1/ace_data_test_labels.pkl', 'rb')
ace_data_test_labels = pickle.load(ace_data_test_labels_file)

ace_data_train_file.close()
ace_data_train_labels_file.close()
ace_data_test_file.close()
ace_data_test_labels_file.close()

# 参数
learningRate = 0.001
training_iters = 320000
batchSize = 1

nInput = 200  # we want the input to take the 28 pixels
nSteps = 1  # every 28
nHidden = 128  # number of neurons for the RNN 64 128 256
nClasses = 34  # this is MNIST so you know

x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])

seq_num = tf.Variable(0)

# weights = tf.Variable(tf.random_normal([2 * nHidden, nClasses]))
# biases = tf.Variable(tf.random_normal([nClasses]))

# 随机初始化每一层的权值和偏置
weights = {
    # Hidden layer weights
    'hidden': tf.Variable(tf.random_normal([2 * nHidden, nInput])),
    'out': tf.Variable(tf.random_normal([nClasses, 2 * nInput]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([nInput])),
    'out': tf.Variable(tf.random_normal([nClasses, 1]))
}


def test_nn(seq_num, x, weights, biases):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, nInput])
    x = tf.split(0, nSteps, x)

    gru_f_cell = tf.nn.rnn_cell.GRUCell(nHidden)
    gru_b_cell = tf.nn.rnn_cell.GRUCell(nHidden)

    outputs, _, _ = tf.nn.bidirectional_rnn(
        gru_f_cell, gru_b_cell, x, dtype=tf.float32)

    yi1 = tf.tanh(tf.matmul(outputs[-1], weights['hidden']) + biases['hidden'])

    yi_1 = []
    yi1_fw = yi1[:seq_num]
    yi1_fw_tensor = tf.argmax(yi1_fw, 0)
    yi1_fw_transpose = tf.transpose(yi1_fw)
    for i in range(nInput):
        yi_1.append(yi1_fw_transpose[i][tf.cast(yi1_fw_tensor[i], tf.int32)])

    yi_2 = []
    yi2_bw = yi1[seq_num:]
    yi2_bw_tensor = tf.argmax(yi2_bw, 0)
    yi2_bw_transpose = tf.transpose(yi2_bw)
    for i in range(nInput):
        yi_2.append(yi2_bw_transpose[i][tf.cast(yi2_bw_tensor[i], tf.int32)])

    y2 = tf.reshape([yi_1, yi_2], [-1, 1])

    results = tf.matmul(weights['out'], y2) + biases['out']
    return results
#     return tf.transpose(results,[-1,1])
    #return tf.reshape(results, [-1, 1])

#     yi1_bw = yi1[seq_num:]
#
#     y_temp = tf.transpose(yi)
#
#     yi_2 = []
#
#     y1_tensor = tf.argmax(yi[:seq_num], 0)
#     y2_tensor = tf.argmax(yi[seq_num:], 0)
#
#     for i in range(nInput):
#         yi_1.append(y_temp[i][tf.cast(y1_tensor[i], tf.int32)])
#         yi_2.append(y_temp[i][tf.cast(y2_tensor[i], tf.int32)])
#         yi_1.append(y_temp[i][y1_tensor[i]])
#         yi_2.append(y_temp[i][y2_tensor[i]])

#     batch_len=len(yi)
#     for i in range(batch_len):
#         yi[i]

    #results = tf.matmul(yi, weights['out']) + biases['out']

    # return tf.transpose(yi1), yi1_fw_tensor, yi2_bw_tensor, yi_1, yi_2, y2


pred = test_nn(seq_num, x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

# def gru_RNN(x, weights, biases):
#     x = tf.transpose(x, [1, 0, 2])
#     x = tf.reshape(x, [-1, nInput])
#     x = tf.split(0, nSteps, x)
#
#     # Define gru cells with tensorflow
#     # Forward direction cell
#     gru_fw_cell = tf.nn.rnn_cell.GRUCell(nHidden)
#     # Backward direction cell
#     gru_bw_cell = tf.nn.rnn_cell.GRUCell(nHidden)
#
#     # Get gru cell output
#     # batch_szie*256
#     outputs, _, _ = tf.nn.bidirectional_rnn(gru_fw_cell, gru_bw_cell, x,dtype=tf.float32)
#
#     results = tf.matmul(outputs[-1], weights) + biases
#     #outputs, states = tf.nn.rnn(gruCell, x, dtype=tf.float32)
#     return results
#
# pred = gru_RNN(x, weights, biases)
#
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

# correctPred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    k = 0

    while k < training_iters:
        step = k % 1600

        batch_xs = ace_data_train[step]
        batch_ys = ace_data_train_labels[step]
        batch_size = len(batch_xs)
        batch_xs = batch_xs.reshape([batch_size, nSteps, nInput])

        for i in range(batch_size):

            if i != 0 and i != batch_size - 1:
                labels = batch_ys[i]
                labels=[labels]
                print(labels)
                labels = np.array(labels).reshape((34, 1)).tolist()
                print(labels)
                print(sess.run(pred,feed_dict={seq_num: i, x: batch_xs}))
                sess.run(cost, feed_dict={seq_num: i, x: batch_xs, y: labels})

        # print('batch_size:'+str(batch_size))
#         print(batch_size)
#
#         results = sess.run(pred, feed_dict={seq_num: 1, x: batch_xs})
#         print(len(results))
#         print(len(results[0]))

#         print(y_temp)
#         print(y1_tensor)
#         print(y2_tensor)
#         print(yi_1)
#         print(yi_2)
#         print(y2)
#         print(len(y_temp))
#         print(yi_1)
#         print(len(yi_1))
# #         print(len(yi_1[0]))
#         print(yi_2)
#         print(len(yi_2))
#         print(len(yi_2[0]))

#         sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

#         for i in range(batch_size):
#             yi_1 = sess.run(pred, feed_dict={seq_num: i, x: batch_xs})
#             print('output_size:' + str(len(yi_1)))
#             print(yi_1)
#             print('--------------------')
        #print('output_size1:' + str(len(yi[0])))


#         print(sess.run(tf.argmax(y, 1), feed_dict={y: batch_ys}))
#         print(sess.run(tf.argmax(pred, 1), feed_dict={x: batch_xs}))
#         print('output_size:' + str(len(yi)))
#         print('output_size1:' + str(len(yi[0])))
        # print(yi)
#         print('output_size2:'+str(len(output[0][0])))

        print(str(k) + '------------------------------------------')
        #sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})


#        if step % 100 == 0:
#             sk = 0
#             acck = 0
#             predictionk, y_k = sess.run([tf.argmax(pred, 1), tf.argmax(y, 1)], feed_dict={x: batch_xs, y: batch_ys})
#             for t in range(len(y_k)):
#                 if y_k[t] != 33:
#                     #                     logging.info(
#                     #                         'actual:' + str(y_k[t]) + '\t predict:' + str(predictionk[t]))
#                     #print(str(y_[t]) + '\t' + str(prediction[t]))
#                     sk = sk + 1
#                     if y_k[t] == predictionk[t]:
#                         acck = acck + 1
#
#             if sk != 0:
#                 print("Iter " + str(k) + '-----------acc=' + str(acck / sk))
        k += 1
    print('Optimization finished')

#     # 载入测试集进行测试
#     length = len(ace_data_test)
#     test_accuracy = 0.0
#     p_s = 0  # 识别的个体总数
#     r_s = 0  # 测试集中存在个个体总数
#     pr_acc = 0  # 正确识别的个数
# #     s = 0
# #     acc = 0
#     for i in range(length):
#         test_len = len(ace_data_test[i])
#         test_data = ace_data_test[i].reshape(
#             (-1, nSteps, nInput))  # 8
#         test_label = ace_data_test_labels[i]
#         # prediction识别出的结果，y_测试集中的正确结果
#         prediction, y_ = sess.run(
#             [tf.argmax(pred, 1), tf.argmax(y, 1)], feed_dict={x: test_data, y: test_label})
#         for t in range(len(y_)):
#             if prediction[t] != 33:
#                 p_s = p_s + 1
#
#             if y_[t] != 33:
#                 r_s = r_s + 1
#                 if y_[t] == prediction[t]:
#                     pr_acc = pr_acc + 1
#
#     print('----------------------------------------------------')
#     print(str(pr_acc) + '------------' + str(r_s))
#     p = pr_acc / p_s
#     r = pr_acc / r_s
#     f = 2 * p * r / (p + r)
#     print('P=' + str(p) + "\tR=" + str(r) + "\tF=" + str(f))


# 299------------579
# P=0.5567970204841713    R=0.5164075993091537    F=0.5358422939068099


# 312------------579
# P=0.5397923875432526    R=0.538860103626943    F=0.5393258426966293
