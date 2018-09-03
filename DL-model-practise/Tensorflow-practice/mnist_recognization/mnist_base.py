import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# import os
# # os.environ['TF_CPP_MIN_LOG_LEVEL']='3' #Disable Tensorflow debugging information
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #Set GPU device -1 -> set cpu to operate

if __name__ == "__main__":
    x = tf.placeholder(tf.float32,[None, 784])
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x, W) + b)

    y_ = tf.placeholder(tf.float32,[None,10])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(cross_entropy)

    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))