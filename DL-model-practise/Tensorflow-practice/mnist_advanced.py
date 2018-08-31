import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

# import os
# # os.environ['TF_CPP_MIN_LOG_LEVEL']='3' #Disable Tensorflow debugging information
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #Set GPU device -1 -> set cpu to operate

def weight_variable(shape):
    initial = tf.truncated_normal(shape, mean=0.0, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

if __name__ == "__main__":

    t_start = time.clock()
    x = tf.placeholder(tf.float32, [None, 784])

    y_target = tf.placeholder(tf.float32, [None, 10])

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # -1 here is the number undetermined
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # get the picture with dimension of 28*28*32, where 32 is the the number of feature(kernel)
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # pooling and transform the dimension to 14*14*32
    h_pool1 = max_pool_2x2(h_conv1)

    # get the picture with dimension of 14*14*64
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # pooling and transform the dimension to 7*7*64
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    # transform the three-dimensional pic to one dimension version of 7*7*64
    # -1 here is the same to the previous, which means the undetermined number of batch size
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    # strictly specified scale of matrix can be multiplied
    # that is (n * 7*7*64)  *  (7*7*64 * 1024)  ->   n * 1024
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # add the Dropout Layer to the pic with two dimension of n * 1024
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    # output layer with two dimension of n * 10
    y_predict = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = -tf.reduce_sum(y_target * tf.log(y_predict))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train = optimizer.minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:

        # save log for tensorboard
        writer = tf.summary.FileWriter("logs/", sess.graph)

        init = tf.initialize_all_variables()
        sess.run(init)

        for i in range(20000):
            x_train, y_train = mnist.train.next_batch(50)
            train.run(feed_dict={x: x_train, y_target: y_train, keep_prob: 0.5})

            if i % 100 == 0:
                x_test, y_test = mnist.test.images, mnist.test.labels
                # set kept-rate to 1.0 in Dropout layer when the NN is tested
                nw_accuracy = accuracy.eval(feed_dict={x: x_test, y_target: y_test, keep_prob: 1.0})
                print("step %d, training accuracy: %.6f" % (i, nw_accuracy))

    t_end = time.clock()
    print("running time:", (t_end - t_start))
