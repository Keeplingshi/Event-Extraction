import numpy as np
import tensorflow as tf

def conv2d_test():
    #Input: x
    x_image = tf.placeholder(tf.float32,shape=[5,5])
    x = tf.reshape(x_image,[1,5,5,1])

    #Filter: W
    W_cpu = np.array([[1,1,1],[0,-1,0],[0,-1,1]],dtype=np.float32)
    W = tf.Variable(W_cpu)
    W = tf.reshape(W, [3,3,1,1])

    #Stride & Padding
    strides=[1, 1, 1, 1]
    padding='VALID'

    #Convolution
    y = tf.nn.conv2d(x, W, strides, padding)

    x_data = np.array([ [1,0,0,0,0]
                       ,[2,1,1,2,1]
                       ,[1,1,2,2,0]
                       ,[2,2,1,0,0]
                       ,[2,1,2,1,1]],dtype=np.float32)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        x = (sess.run(x, feed_dict={x_image: x_data}))
        W = (sess.run(W, feed_dict={x_image: x_data}))
        y = (sess.run(y, feed_dict={x_image: x_data}))

        print("The shape of x:\t", x.shape, ",\t and the x.reshape(5,5) is :")
        print(x.reshape(5,5))

        print("The shape of x:\t", W.shape, ",\t and the W.reshape(3,3) is :")
        print(W.reshape(3,3))

        print("The shape of y:\t", y.shape, ",\t and the y.reshape(3,3) is :")
        print(y.reshape(3,3))




def max_pool_test(x_data):
    (width, height) = x_data.shape

    #input:x
    x_image = tf.placeholder(tf.float32,shape=[width,height])
    x = tf.reshape(x_image,[1,width,height,1])

    #max pooling
    y = tf.nn.max_pool(x,[1,1,height,1],[1,1,1,1],'VALID')

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        x = (sess.run(x,feed_dict={x_image:x_data}))
        y = (sess.run(y,feed_dict={x_image:x_data}))

        print("The shape of x:\t", x.shape, ",\t and the x.reshpe(4,4) is :")
        print(x.reshape(width,height))

        print("The shape of y:\t", y.shape, ",\t and the y.reshpe(2,2) is :")
        (a,b,c,d)=y.shape
        print(y.reshape(b,c))


conv_size=2
x_data = tf.placeholder(tf.float32, shape=[None, None])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

W_conv1 = weight_variable([conv_size,conv_size,1,1])
b_conv1 = bias_variable([1])

def conv2d_max_pool(x_data):
    #cnn 卷积和池化
    width=tf.shape(x_data)[0]
    height=tf.shape(x_data)[1]
    x=tf.reshape(x_data,[1,width, height,1])

    #Filter: W
    # W_cpu = np.array([[1,1,1],[1,1,1],[1,1,1]],dtype=np.float32)
    # W = tf.Variable(W_cpu)
    # W = tf.reshape(W, [3,3,1,1])

    #Convolution  Stride & Padding
    con2d_result = tf.nn.conv2d(x, W_conv1, [1, 1, 1, 1], 'VALID')
    con2d_result=tf.reshape(con2d_result,[1,width-conv_size+1,height-conv_size+1,1])

    h_conv1=tf.nn.relu(con2d_result+b_conv1)

    max_pool_result = tf.nn.max_pool(h_conv1, [1,500,1,1], [1,1,1,1], 'SAME')
    max_pool_result=tf.reshape(max_pool_result,[width-conv_size+1,height-conv_size+1])[0]

    return x,con2d_result,max_pool_result,h_conv1
    #

test=conv2d_max_pool(x_data)

if __name__ == '__main__':

    x_tmp=[i for i in range(55)]
    x_tmp = np.array([x_tmp],dtype=np.float32).reshape(11, 5)
    # print(x_tmp)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        w=sess.run(W_conv1)
        # print(np.array(w).shape)

        x_data,con2d_result,max_pool_result,h_conv1=sess.run(test, feed_dict={x_data: x_tmp})
        print(np.array(x_data).shape)
        print(np.array(con2d_result).shape)
        print(np.array(max_pool_result).shape)
        # print(np.array(con2d_result).reshape(9,3))

        print(np.array(h_conv1).shape)
        print(np.array(max_pool_result))
        # x_data,x, con2d_result, max_pool_result=sess.run(test, feed_dict={x_data: x_tmp})
        # print(np.array(x_data).shape)
        # print(np.array(x).shape)
        # print(np.array(con2d_result).shape)
        # print(np.array(max_pool_result).shape)
        #
        # print(con2d_result)
        # print(max_pool_result)



    # print('-------------------conv2d_test--------------------')
    # conv2d_test()
    #

    # x_tmp=[i+1 for i in range(2682)]
    # x_data = np.array([x_tmp],dtype=np.float32).reshape(298, 9)
    # x_tmp=[i+1 for i in range(28)]
    # x_data = np.array([x_tmp],dtype=np.float32).reshape(4, 7)
    # print(x_data.shape)
    # print('------------------max_pool_test---------------------')
    # max_pool_test(x_data)

