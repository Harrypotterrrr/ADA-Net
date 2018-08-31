import tensorflow as tf
from add_layers import add_layers
import numpy as np

eps = 1e-2


def f(x):
    return np.square(x + 1.0) - 2.0

if __name__ == "__main__":

    x_sample = tf.placeholder(tf.float32, [None, 1])
    y_sample = tf.placeholder(tf.float32, [None, 1])

    hide_layer1 = add_layers(x_sample, 1, 100, activation_function=None)
    hide_layer2 = add_layers(hide_layer1, 100, 100, activation_function=tf.nn.relu)
    y_predict = add_layers(hide_layer2, 100, 1, activation_function=None)

    error_sum = tf.reduce_sum(tf.square(y_sample - y_predict), reduction_indices=[1])
    loss = tf.reduce_mean(error_sum)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(1000):
            # x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
            x_data = np.random.normal(0, 1, [300, 1])
            noise = np.random.normal(0, 0.05, x_data.shape)
            y_data = f(x_data) + noise

            sess.run(train, feed_dict={x_sample: x_data, y_sample: y_data})

            if i % 20 == 0:
                print(sess.run(loss, feed_dict={x_sample:x_data, y_sample : y_data}))

        x_test = np.random.uniform(-1, 1, [1000, 1])
        y_test = f(x_test)

        y_get = sess.run(y_predict, feed_dict={x_sample:x_test})

        sum = 0
        for i in range(0,x_test.shape[0]):
            if abs(y_test[i] - y_get[i]) < eps:
                sum += 1
        print("sum:",sum,"\n===================")

        for i in np.random.randint(0,1000,500):
            print("i: ",i)
            print("x_test: ",x_test[i])
            print("y_test,y_predict:")
            print(y_test[i],y_get[i])

        accuracy = tf.reduce_mean(tf.cast(abs(tf.add(y_sample, -y_predict)) < eps, tf.float32))
        print(accuracy.eval(feed_dict={y_sample: y_test, x_sample: x_test}))
