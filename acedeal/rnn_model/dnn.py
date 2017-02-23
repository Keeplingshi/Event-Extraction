import tensorflow as tf,pickle,time

# Load data
print("Loading data...")
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
sess = tf.InteractiveSession()
x_ = tf.placeholder("float",shape = [None,200])
y_ = tf.placeholder("float",shape = [None,34])


W1 = tf.Variable(tf.random_uniform([200,100],minval=-2.0,maxval=2.0),name="W1")
b1 = tf.Variable(tf.zeros([100]),name="b1")
W2 = tf.Variable(tf.random_uniform([100,50],minval=-2.0,maxval=2.0),name="W2")
b2 = tf.Variable(tf.zeros([50]),name="b2")
W3 = tf.Variable(tf.random_uniform([50,34],minval=-2.0,maxval=2.0),name="W3")
b3 = tf.Variable(tf.zeros([34]),name="b3")

# Add ops to save and restore all the variables.
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
# saver.restore(sess, "C:/Users/Administrator.SC-201609131415/PycharmProjects/untitled/model/DNN2W1.ckpt")

y1 = tf.nn.softmax(tf.matmul(x_,W1)+b1)
y2 = tf.nn.softmax(tf.matmul(y1,W2)+b2)
y3 = tf.nn.softmax(tf.matmul(y2,W3)+b3)


# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y3, y_))
cost = -tf.reduce_sum(y_*tf.log(y3))
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# cross_entropy = -tf.reduce_sum(y_*tf.log(y3))
# weight = [[0.06],[1.98],[0.98],[0.98]]
# cross_entropy = -tf.reduce_sum(tf.matmul(y_*tf.log(y3),weight))

#train_step = tf.train.GradientDescentOptimizer(0.008).minimize(cross_entropy)


if __name__ == "__main__":
    # 数据读取，训练集和测试集
    ace_data_train_file = open('../ace_data_process/ace_data6/ace_data_train.pkl', 'rb')
    ace_data_train = pickle.load(ace_data_train_file)
     
    ace_data_train_labels_file = open('../ace_data_process/ace_data6/ace_data_train_labels.pkl', 'rb')
    ace_data_train_labels = pickle.load(ace_data_train_labels_file)
     
    ace_data_test_file = open('../ace_data_process/ace_data6/ace_data_test.pkl', 'rb')
    ace_data_test = pickle.load(ace_data_test_file)
     
    ace_data_test_labels_file = open('../ace_data_process/ace_data6/ace_data_test_labels.pkl', 'rb')
    ace_data_test_labels = pickle.load(ace_data_test_labels_file)
    
    batch_size=20
    data_len=len(ace_data_train)
    for j in range(10000):
        batch_start=batch_size*j
        batch_end=batch_start+batch_size
        
        optimizer.run(feed_dict={x_: ace_data_train[batch_start:batch_end], y_: ace_data_train_labels[batch_start:batch_end]})
        if(j%10==0):
            correct_prediction = tf.equal(tf.argmax(y3, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            
            print("Setp: ",j, "Accuracy: ", sess.run(accuracy, feed_dict={x_: ace_data_test, y_: ace_data_test_labels}))
        
        
    print('Optimization finished')

    # 载入测试集进行测试
    length = len(ace_data_test)
    test_accuracy = 0.0
    p_s = 0  # 识别的个体总数
    r_s = 0  # 测试集中存在个个体总数
    pr_acc = 0  # 正确识别的个数

    prediction, y_ = sess.run([tf.argmax(y3, 1), tf.argmax(y_, 1)], feed_dict={x_: ace_data_test, y_: ace_data_test_labels})
    
    print(prediction)
    print(len(prediction))
    print(y_)
    print(len(y_))
    
    for t in range(len(y_)):
        if prediction[t] != 33:
            p_s = p_s + 1
 
        if y_[t] != 33:
            r_s = r_s + 1
            if y_[t] == prediction[t]:
                pr_acc = pr_acc + 1
#     for i in range(length):
#         # prediction识别出的结果，y_测试集中的正确结果
#         prediction, y_ = sess.run([tf.argmax(y3, 1), tf.argmax(y_, 1)], feed_dict={x_: ace_data_test, y_: ace_data_test_labels})
#         for t in range(len(y_)):
#             if prediction[t] != 33:
#                 p_s = p_s + 1
# 
#             if y_[t] != 33:
#                 r_s = r_s + 1
#                 if y_[t] == prediction[t]:
#                     pr_acc = pr_acc + 1

    print('----------------------------------------------------')
    print(str(pr_acc) + '------------' + str(r_s))
    p = pr_acc / p_s
    r = pr_acc / r_s
    f = 2 * p * r / (p + r)
    print('P=' + str(p) + "\tR=" + str(r) + "\tF=" + str(f))
# dataFile = open('..//ace_data_process/ace_data6/ace_data_train.pkl', 'rb')
# labelFile = open('..//ace_data_process/ace_data6/ace_data_train_labels.pkl', 'rb')
# xI,yI = pickle.load(dataFile),pickle.load(labelFile)
# for j in range(100):
#     optimizer.run(feed_dict={x_: xI[i:i+500], y_: yI})
#     if(j%1==0):
#         correct_prediction = tf.equal(tf.argmax(y3, 1), tf.argmax(y_, 1))
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#         testData = open('..//ace_data_process/ace_data6/ace_data_test.pkl', 'rb')
#         testLabel = open('..//ace_data_process/ace_data6/ace_data_test_labels.pkl', 'rb')
#         batch = [pickle.load(testData),pickle.load(testLabel)]
#         print("Setp: ", "Accuracy: ", sess.run(accuracy, feed_dict={x_: batch[0], y_: batch[1]}))
# #     save_path = saver.save(sess, "C:/Users/Administrator.SC-201609131415/PycharmProjects/untitled/model/DNN2W1.ckpt")
# #     print ("Model saved in file: ", save_path)
# print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))