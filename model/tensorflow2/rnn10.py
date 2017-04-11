from __future__ import print_function
import tensorflow as tf
import numpy as np
import argparse,pickle
import sys


class Model:
    def __init__(self, args):
        self.args = args
        self.input_data = tf.placeholder(tf.float32, [None, args.sentence_length, args.word_dim])
        self.output_data = tf.placeholder(tf.float32, [None, args.sentence_length, args.class_size])
        self.input_length=tf.placeholder(tf.int64, [None])

        #cnn process
        cnn_weight = self.cnn_weight_variable([3,3,1,1])
        cnn_bias=self.cnn_bias_variable([1])
        cnn_output=self.cnn_conv2d_max_pool(self.input_data,args,cnn_weight,cnn_bias)
        cnn_output=tf.reshape(cnn_output,[-1,args.word_dim])
        cnn_extend=[]
        for i in range(args.sentence_length):
            cnn_extend.append(cnn_output)

        #lstm process
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(args.hidden_layers, state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(args.hidden_layers, state_is_tuple=True)

        self.length=tf.cast(self.input_length, tf.int32)
        output, _,_ = tf.nn.bidirectional_rnn(fw_cell, bw_cell,
                                               tf.unpack(tf.transpose(self.input_data, perm=[1, 0, 2])),
                                               dtype=tf.float32, sequence_length=self.length)
        #cnn lstm contact
        lstm_cnn_output=tf.concat(2,[output,cnn_extend])

        weight, bias = self.weight_and_bias(2 * args.hidden_layers+args.word_dim, args.class_size)
        output = tf.reshape(tf.transpose(tf.pack(lstm_cnn_output), perm=[1, 0, 2]), [-1, 2 * args.hidden_layers+args.word_dim])


        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        self.prediction = tf.reshape(prediction, [-1, args.sentence_length, args.class_size])
        self.loss = self.cost()
        optimizer = tf.train.AdamOptimizer(args.learning_rate)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 10)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def cost(self):
        cross_entropy = self.output_data * tf.log(self.prediction)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        mask = tf.sign(tf.reduce_max(tf.abs(self.output_data), reduction_indices=2))
        cross_entropy *= mask
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(cross_entropy)

    @staticmethod
    def weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

    @staticmethod
    def cnn_conv2d_max_pool(data,args,cnn_weight,cnn_bias):
        x = tf.reshape(data, [-1,args.sentence_length,args.word_dim,1])
        conv1=tf.nn.conv2d(x, cnn_weight, strides=[1,1,1,1], padding='SAME')
        h_conv1 = tf.nn.relu(conv1 + cnn_bias)
        max_pool1=tf.nn.max_pool(h_conv1, ksize=[1,args.sentence_length,1,1], strides=[1,1,1,1], padding='VALID')
        return max_pool1

    @staticmethod
    def cnn_weight_variable(shape):
        weight = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(weight)

    @staticmethod
    def cnn_bias_variable(shape):
        bias = tf.constant(0.1, shape=shape)
        return tf.Variable(bias)


def f1(prediction, target, length,iter):

    prediction = np.argmax(prediction, 2)
    target = np.argmax(target, 2)

    iden_p=0   # 识别的个体总数
    iden_r=0    # 测试集中存在个个体总数
    iden_acc=0  # 正确识别的个数

    classify_p = 0  # 识别的个体总数
    classify_r = 0  # 测试集中存在个个体总数
    classify_acc = 0  # 正确识别的个数

    for i in range(len(target)):
        for j in range(length[i]):
            if prediction[i][j]!=0:
                classify_p+=1
                iden_p+=1

            if target[i][j]!=0:
                classify_r+=1
                iden_r+=1

            if target[i][j]==prediction[i][j] and target[i][j]!=0:
                classify_acc+=1

            if prediction[i][j]!=0 and target[i][j]!=0:
                iden_acc+=1

    try:
        print('-----------------------' + str(iter) + '-----------------------------')
        print('Trigger Identification:')
        print(str(iden_acc) + '------' + str(iden_p) + '------' + str(iden_r))
        p = iden_acc / iden_p
        r = iden_acc / iden_r
        if p + r != 0:
            f = 2 * p * r / (p + r)
            print('P=' + str(p) + "\tR=" + str(r) + "\tF=" + str(f))
        print('Trigger Classification:')
        print(str(classify_acc) + '------' + str(classify_p) + '------' + str(classify_r))
        p = classify_acc / classify_p
        r = classify_acc / classify_r
        if p + r != 0:
            f = 2 * p * r / (p + r)
            print('P=' + str(p) + "\tR=" + str(r) + "\tF=" + str(f))
            print('------------------------' + str(iter) + '----------------------------')
            return f
    except ZeroDivisionError:
        print('-----------------------' + str(iter) + '-----------------------------')
        print('all zero')
        print('-----------------------' + str(iter) + '-----------------------------')
        return 0


def train(args):
    saver_path="./data/saver/checkpointrnn10_1.data"

    data_f = open('./data/3/train_data_form34.data', 'rb')
    X_train,Y_train,W_train,L_train,X_test,Y_test,W_test,L_test,X_dev,Y_dev,W_dev,L_dev = pickle.load(data_f)
    data_f.close()
    train_inp=X_train
    train_out=Y_train
    test_a_inp=X_test
    test_a_out=Y_test

    model = Model(args)
    maximum = 0
    with tf.Session() as sess:
#        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, saver_path)

        pred, length = sess.run([model.prediction, model.length]
                                    , {model.input_data: test_a_inp,model.output_data: test_a_out,model.input_length:L_test})

        maximum=f1(pred, test_a_out, length,1)
        sys.exit()

        # train_inp=train_inp[:84]
        # train_out=train_out[:84]
        # print(np.array(train_inp).shape)
        for e in range(args.epoch):
            for ptr in range(0, len(train_inp), args.batch_size):

                sess.run(model.train_op, {model.input_data: train_inp[ptr:ptr + args.batch_size]
                    ,model.output_data: train_out[ptr:ptr + args.batch_size]
                    ,model.input_length:L_train[ptr:ptr + args.batch_size]})


            pred, length = sess.run([model.prediction, model.length]
                                    , {model.input_data: test_a_inp,model.output_data: test_a_out,model.input_length:L_test})

            m = f1(pred, test_a_out, length,e)
            if m>maximum:
                saver = tf.train.Saver(tf.global_variables())
                saver.save(sess,saver_path)
                maximum=m




parser = argparse.ArgumentParser()
parser.add_argument('--word_dim', type=int,default=300, help='dimension of word vector')
parser.add_argument('--sentence_length', type=int,default=60, help='max sentence length')
parser.add_argument('--class_size', type=int, default=34,help='number of classes')
parser.add_argument('--learning_rate', type=float, default=0.003,help='learning_rate')
parser.add_argument('--hidden_layers', type=int, default=128, help='hidden dimension of rnn')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers in rnn')
parser.add_argument('--batch_size', type=int, default=100, help='batch size of training')
parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--restore', type=str, default=None, help="path of saved model")
train(parser.parse_args())
