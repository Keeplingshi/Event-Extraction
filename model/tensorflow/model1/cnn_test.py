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


def max_pool_test():
    #input:x
    x_image = tf.placeholder(tf.float32,shape=[4,4])
    x = tf.reshape(x_image,[1,4,4,1])

    #ksize & Stride & padding
    ksize = [1,2,2,1]
    strides=[1,2,2,1]
    padding = 'VALID'

    #max pooling
    y = tf.nn.max_pool(x,ksize,strides,padding)

    x_data = np.array([    [4,3,1,8]
                          ,[7,2,6,3]
                          ,[2,0,1,1]
                          ,[3,4,2,5]],dtype = np.float32)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        x = (sess.run(x,feed_dict={x_image:x_data}))
        y = (sess.run(y,feed_dict={x_image:x_data}))

        print("The shape of x:\t", x.shape, ",\t and the x.reshpe(4,4) is :")
        print(x.reshape(4,4))

        print("The shape of y:\t", y.shape, ",\t and the y.reshpe(2,2) is :")
        print(y.reshape(2,2))


if __name__ == '__main__':
    print('-------------------conv2d_test--------------------')
    conv2d_test()

    print('------------------max_pool_test---------------------')
    max_pool_test()

