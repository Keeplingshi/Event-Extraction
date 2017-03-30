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


# (width, height) = x_data.shape

# width=3
# height=4
rr=1
# x_image = tf.placeholder(tf.float32, shape=[width, height])
x_data = tf.placeholder(tf.float32, shape=[None, None])
# width=tf.placeholder(tf.int64, [0])
tt=tf.placeholder(tf.int64, [0])

def conv2d_max_pool(x_data):
    # width = tf.cast(width, dtype=tf.int32)
    # height = tf.cast(height, dtype=tf.int32)
    m=tf.shape(x_data)
    width=m[0]
    height=m[1]
    x = tf.reshape(x_data, [1, width, height, 1])

    #Filter: W
    W_cpu = np.array([[1,1,1],[1,1,1],[1,1,1]],dtype=np.float32)
    W = tf.Variable(W_cpu)
    W = tf.reshape(W, [3,3,1,1])

    #Convolution  Stride & Padding
    con2d_result = tf.nn.conv2d(x, W, [1, 1, 1, 1], 'VALID')
    con2d_result=tf.reshape(con2d_result,[width-2,height-2])

    # max_pool_result = tf.nn.max_pool(con2d_result, [1,1,9,1], [1,1,1,1], 'VALID')
    # max_pool_result=tf.reshape(max_pool_result,[width-2])

    return con2d_result,width,height

test=conv2d_max_pool(x_data)

    # with tf.Session() as sess:
    #     init = tf.global_variables_initializer()
    #     sess.run(init)
    #
    #     x = (sess.run(x, feed_dict={x_image: x_data}))
    #     W = (sess.run(W, feed_dict={x_image: x_data}))
    #     con2d_result = (sess.run(con2d_result, feed_dict={x_image: x_data}))
    #     max_pool_result= (sess.run(max_pool_result, feed_dict={x_image: x_data}))
    #
    #     print("The shape of x:\t", x.shape, ",\t and the x.reshape(5,5) is :")
    #     print(x.reshape(width,height))
    #
    #     print("The shape of x:\t", W.shape, ",\t and the W.reshape(3,3) is :")
    #     print(W.reshape(3,3))
    #
    #     print("The shape of y:\t", con2d_result.shape, ",\t and the y.reshape is :")
    #     print(con2d_result.reshape(width-2,height-2))
    #
    #     print("The shape of y:\t", max_pool_result.shape, ",\t and the y.reshape is :")
    #     # (a, b, c, d) = max_pool_result.shape
    #     print(max_pool_result)
    #
    # return 0


if __name__ == '__main__':

    # x=[i for i in range(3300)]
    # x=np.array(x).reshape(300,11)
    # print(x.shape)
    # print(x)
    # print('----------------------------------------')

    x_tmp=[i+1 for i in range(3300)]
    x_tmp = np.array([x_tmp],dtype=np.float32).reshape(300, 11)
    print(x_tmp.shape)
    # (a,b)=x_tmp.shape
    # print(a,b)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        result,a,b=sess.run(test, feed_dict={x_data: x_tmp})
        print(result)
        print(a)
        print(b)
        print(result.shape)



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

