from __future__ import print_function
import tensorflow as tf
import numpy as np
import argparse,pickle
import sys


class Model:
    def __init__(self, args):
        self.args = args
        self.input_data = tf.placeholder(tf.float32, [None, args.sentence_length, args.word_dim+args.word_dist])
        self.output_data = tf.placeholder(tf.float32, [None, args.sentence_length, args.class_size])

        self.x_front = tf.slice(self.input_data, [0, 0, 0], [-1, -1, args.word_dim])
        x_posi = tf.slice(self.input_data, [0, 0, args.word_dim], [-1, -1, args.word_dist])

        #cnn process
        cnn_weight = self.cnn_weight_variable([args.filter_size,args.word_dim,1,args.feature_maps])
        cnn_bias=self.cnn_bias_variable([args.feature_maps])
        self.cnn_output=self.cnn_conv2d_max_pool(self.x_front,args,cnn_weight,cnn_bias)
        self.cnn_output=tf.reshape(tf.transpose(self.cnn_output,[1,0,2,3]), [args.sentence_length, args.batch_size,args.feature_maps])

        #lstm process
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(args.hidden_layers, state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(args.hidden_layers, state_is_tuple=True)

        fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=0.5)
        bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=0.5)

        fw_cell = tf.nn.rnn_cell.MultiRNNCell([fw_cell] * args.num_layers, state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.MultiRNNCell([bw_cell] * args.num_layers, state_is_tuple=True)


        used = tf.sign(tf.reduce_max(tf.abs(self.input_data), reduction_indices=2))
        self.length = tf.cast(tf.reduce_sum(used, reduction_indices=1), tf.int32)
        output, _,_ = tf.nn.bidirectional_rnn(fw_cell, bw_cell,
                                               tf.unpack(tf.transpose(self.x_front, perm=[1, 0, 2])),
                                               dtype=tf.float32, sequence_length=self.length)

        self.posi=tf.transpose(x_posi,[1,0,2])
        #cnn lstm contact
        self.lstm_cnn_output=tf.concat(2,[output,self.cnn_output,self.posi])

        weight_x=2 * args.hidden_layers+args.feature_maps+args.word_dist
        weight, bias = self.weight_and_bias(weight_x, args.class_size)
        output = tf.reshape(tf.transpose(tf.pack(self.lstm_cnn_output), perm=[1, 0, 2]), [-1, weight_x])

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
        pad_seqs = []
        pad_seq=[]
        pad_len=int((args.filter_size-1)/2)
        zero_seq=[0.0 for j in range(args.word_dim)]
        for i in range(pad_len):
            pad_seq.append(zero_seq)
        for i in range(args.batch_size):
            pad_seqs.append(pad_seq)
        conv_pad = tf.reshape(pad_seqs, [args.batch_size, pad_len,args.word_dim,1])
        x = tf.reshape(data, [-1,args.sentence_length,args.word_dim,1])
        x=tf.concat(1,[conv_pad,x,conv_pad])
        conv1=tf.nn.conv2d(x, cnn_weight, strides=[1,1,1,1], padding='VALID')
        h_conv1 = tf.nn.sigmoid(conv1 + cnn_bias)
        return h_conv1

    @staticmethod
    def cnn_weight_variable(shape):
        weight = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(weight)

    @staticmethod
    def cnn_bias_variable(shape):
        bias = tf.constant(0.1, shape=shape)
        return tf.Variable(bias)


def f1(prediction, target, length, iter_num):

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
        print('-----------------------' + str(iter_num) + '-----------------------------')
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
            print('------------------------' + str(iter_num) + '----------------------------')
            return f
    except ZeroDivisionError:
        print('-----------------------' + str(iter_num) + '-----------------------------')
        print('all zero')
        print('-----------------------' + str(iter_num) + '-----------------------------')
        return 0


def train(args):
    saver_path="./data/saver/checkpointrnn12_3.data"

    data_f = open('./data/2/train_data_posi_form34.data', 'rb')
    X_train,Y_train,W_train,X_test,Y_test,W_test,X_dev,Y_dev,W_dev = pickle.load(data_f)
    data_f.close()
    train_inp=X_train
    train_out=Y_train
    test_a_inp=X_test
    test_a_out=Y_test

    model = Model(args)
    maximum = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver = tf.train.Saver(tf.global_variables())
        # saver.restore(sess, saver_path)
        #
        # test_pred = []
        # test_len = []
        # for ptr in range(0, len(test_a_inp), args.batch_size):
        #     batch_xs = test_a_inp[ptr:ptr + args.batch_size]
        #     batch_ys = test_a_out[ptr:ptr + args.batch_size]
        #
        #     if len(batch_xs) < args.batch_size:
        #         batch_xs.extend(test_a_inp[0:args.batch_size - len(batch_xs)])
        #         batch_ys.extend(test_a_out[0:args.batch_size - len(batch_ys)])
        #
        #     pred, length = sess.run([model.prediction, model.length]
        #                             , {model.input_data: batch_xs, model.output_data: batch_ys})
        #     test_pred.extend(pred)
        #     test_len.extend(length)
        #
        # m = f1(test_pred, test_a_out, test_len, 1)
        # maximum=m
        # sys.exit()

        for e in range(args.epoch):
            for ptr in range(0, len(train_inp), args.batch_size):
                batch_xs=train_inp[ptr:ptr + args.batch_size]
                batch_ys=train_out[ptr:ptr + args.batch_size]
                # print(np.array(batch_xs).shape)
                # print(batch_xs[0][0])
                # print(len(batch_xs[0][0]))

                if len(batch_xs)<args.batch_size:
                    batch_xs.extend(train_inp[0:args.batch_size-len(batch_xs)])
                    batch_ys.extend(train_out[0:args.batch_size - len(batch_ys)])

                # x_embed=sess.run(model.x_front, {model.input_data: batch_xs,model.output_data: batch_ys})
                # print(np.array(x_embed).shape)
                # x_posi=sess.run(model.x_posi, {model.input_data: batch_xs,model.output_data: batch_ys})
                # print(np.array(x_posi).shape)
                # x_posi=sess.run(model.lstm_cnn_output, {model.input_data: batch_xs,model.output_data: batch_ys})
                # print(np.array(x_posi).shape)
                # sys.exit()

                sess.run(model.train_op, {model.input_data: batch_xs,model.output_data: batch_ys})

            test_pred=[]
            test_len=[]
            for ptr in range(0, len(test_a_inp), args.batch_size):
                batch_xs=test_a_inp[ptr:ptr + args.batch_size]
                batch_ys=test_a_out[ptr:ptr + args.batch_size]

                if len(batch_xs)<args.batch_size:
                    batch_xs.extend(test_a_inp[0:args.batch_size-len(batch_xs)])
                    batch_ys.extend(test_a_out[0:args.batch_size - len(batch_ys)])

                pred, length = sess.run([model.prediction, model.length]
                                        , {model.input_data: batch_xs, model.output_data: batch_ys})
                test_pred.extend(pred)
                test_len.extend(length)

            m = f1(test_pred, test_a_out, test_len,e)
            if m>maximum:
                saver = tf.train.Saver(tf.global_variables())
                saver.save(sess,saver_path)
                maximum=m




parser = argparse.ArgumentParser()
parser.add_argument('--word_dim', type=int,default=300, help='dimension of word vector')
parser.add_argument('--word_dist', type=int,default=5, help='distance of word in sentence')
parser.add_argument('--sentence_length', type=int,default=60, help='max sentence length')
parser.add_argument('--class_size', type=int, default=34,help='number of classes')
parser.add_argument('--learning_rate', type=float, default=0.003,help='learning_rate')
parser.add_argument('--hidden_layers', type=int, default=100, help='hidden dimension of rnn')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers in rnn')
parser.add_argument('--batch_size', type=int, default=100, help='batch size of training')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs')
parser.add_argument('--restore', type=str, default=None, help="path of saved model")
parser.add_argument('--feature_maps', type=int, default=200, help='feature maps')
parser.add_argument('--filter_size', type=int, default=5, help='conv filter size')
train(parser.parse_args())


