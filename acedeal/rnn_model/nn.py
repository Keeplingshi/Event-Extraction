import tensorflow as tf
import pickle

# 数据读取，训练集和测试集
ace_data_train_file = open('../ace_data_process/ace_data6/ace_data_train.pkl', 'rb')
ace_data_train = pickle.load(ace_data_train_file)
     
ace_data_train_labels_file = open('../ace_data_process/ace_data6/ace_data_train_labels.pkl', 'rb')
ace_data_train_labels = pickle.load(ace_data_train_labels_file)
     
ace_data_test_file = open('../ace_data_process/ace_data6/ace_data_test.pkl', 'rb')
ace_data_test = pickle.load(ace_data_test_file)
     
ace_data_test_labels_file = open('../ace_data_process/ace_data6/ace_data_test_labels.pkl', 'rb')
ace_data_test_labels = pickle.load(ace_data_test_labels_file)

# 定义变量
x = tf.placeholder("float", [None, 200])
y_ = tf.placeholder("float",shape = [None,34])
# 权重
W = tf.Variable(tf.zeros([200, 34]))
# 偏置量
b = tf.Variable(tf.zeros([34]))

# 实现模型
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# 梯度下降，最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.004).minimize(cross_entropy)

# 计算正确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 初始化所有变量
init = tf.global_variables_initializer()
# 启动模型
sess = tf.Session()
sess.run(init)

batch_size=100
# 训练模型，循环训练1000次
for i in range(10000):
    batch_start=batch_size*i
    batch_end=batch_start+batch_size
    
    batch_xs=ace_data_train[batch_start:batch_end]
    batch_ys=ace_data_train_labels[batch_start:batch_end]
    
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
    if i%100==0:
        print("Setp: ",i, "Accuracy: ", sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))


print('----------------------------')
print(sess.run(accuracy, feed_dict={x: ace_data_test, y_: ace_data_test_labels}))

