# coding: utf-8
# @author: chenbin
# @date：2017-02-23
# 

# In[1]:

import tensorflow as tf
import numpy as np
import pickle


# In[2]:

sess = tf.InteractiveSession()


# In[3]:
# 数据读取，训练集和测试集
ace_data_train_file = open('../ace_data_process/ace_data7/ace_data_train.pkl', 'rb')
ace_data_train = pickle.load(ace_data_train_file)

ace_data_train_labels_file = open('../ace_data_process/ace_data7/ace_data_train_labels.pkl', 'rb')
ace_data_train_labels = pickle.load(ace_data_train_labels_file)

ace_data_test_file = open('../ace_data_process/ace_data7/ace_data_test.pkl', 'rb')
ace_data_test = pickle.load(ace_data_test_file)

ace_data_test_labels_file = open('../ace_data_process/ace_data7/ace_data_test_labels.pkl', 'rb')
ace_data_test_labels = pickle.load(ace_data_test_labels_file)

ace_data_train_file.close()
ace_data_train_labels_file.close()
ace_data_test_file.close()
ace_data_test_labels_file.close()

# In[4]:

# RNN学习时使用的参数
learning_rate = 0.001  # 1
training_iters = 16000
batch_size = 1

# 神经网络的参数
n_input = 200  # 输入层的n
n_steps = 1  # 28长度
n_hidden = 128  # 隐含层的特征数
n_classes = 34  # 输出的数量，因为是分类问题，这里一共有34个


x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])


# In[5]:


# In[6]:
'''
lstm详细过程
'''
def RNN(x, n_steps, n_input, n_hidden, n_classes):
    # Parameters:
    # Input gate: input, previous output, and bias
    ix = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
    im = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
    ib = tf.Variable(tf.zeros([1, n_hidden]))
    # Forget gate: input, previous output, and bias
    fx = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
    fm = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
    fb = tf.Variable(tf.zeros([1, n_hidden]))
    # Memory cell: input, state, and bias
    cx = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
    cm = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
    cb = tf.Variable(tf.zeros([1, n_hidden]))
    # Output gate: input, previous output, and bias
    ox = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
    om = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
    ob = tf.Variable(tf.zeros([1, n_hidden]))
    # Classifier weights and biases
    w = tf.Variable(tf.truncated_normal([n_hidden, n_classes]))
    b = tf.Variable(tf.zeros([n_classes]))

    # Definition of the cell computation
    def lstm_cell(i, o, state):
        input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
        forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
        update = tf.tanh(tf.matmul(i, cx) + tf.matmul(o, cm) + cb)
        state = forget_gate * state + input_gate * update
        output_gate = tf.sigmoid(tf.matmul(i, ox) +  tf.matmul(o, om) + ob)
        return output_gate * tf.tanh(state), state
    
    # Unrolled LSTM loop
    outputs = list()
    state = tf.Variable(tf.zeros([batch_size, n_hidden]))
    output = tf.Variable(tf.zeros([batch_size, n_hidden]))
    
    # x shape: (batch_size, n_steps, n_input)
    # desired shape: list of n_steps with element shape (batch_size, n_input)
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(0, n_steps, x)
    for i in x:
        output, state = lstm_cell(i, output, state)
        outputs.append(output)
    logits =tf.matmul(outputs[-1], w) + b
    return logits


# In[7]:

pred = RNN(x, n_steps, n_input, n_hidden, n_classes)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

def compute_accuracy(test_data , test_label ):
    
    return tf.argmax(pred, 1)
# correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()


# In[8]:

# Launch the graph
sess.run(init)

k = 0
# 持续迭代
while k < training_iters:
    step = k % 1600
    batch_xs = ace_data_train[step]
    batch_ys = ace_data_train_labels[step]
    batch_size = len(batch_xs)
    batch_xs = batch_xs.reshape([batch_size, n_steps, n_input])
    sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
    
    if k % 100 == 0:
        acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
        loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})
        print("Iter " + str(k) + ", Minibatch Loss= " +               "{:.6f}".format(loss) + ", Training Accuracy= " +               "{:.5f}".format(acc))
    
    k += 1
    
print("Optimization Finished!")

# In[9]:
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
    test_data = ace_data_test[i].reshape((-1, n_steps, n_input))  # 8
    test_label = ace_data_test_labels[i]
    # prediction识别出的结果，y_测试集中的正确结果
    prediction, y_ = sess.run([tf.argmax(pred, 1), tf.argmax(y, 1)], feed_dict={x: test_data, y: test_label})
    for t in range(len(y_)):
        if prediction[t] != 33:
            p_s = p_s + 1

        if y_[t] != 33:
            r_s = r_s + 1
            if y_[t] == prediction[t]:
                pr_acc = pr_acc + 1


print('----------------------------------------------------')
print(str(pr_acc) + '------------' + str(r_s))
p = pr_acc / p_s
r = pr_acc / r_s
f = 2 * p * r / (p + r)
print('P=' + str(p) + "\tR=" + str(r) + "\tF=" + str(f))
