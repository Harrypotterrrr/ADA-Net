import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

eps = 1e-2
batch_size = 500


def f(x):
    return np.square(x + 1.0) - 2.0

def test_model():

    x_test = np.random.uniform(-1, 1, [1000, 1])
    y_test = f(x_test)

    y_prediction = sess.run(outputs, feed_dict={x_sample: x_test})

    writer = tf.summary.FileWriter("logs/", sess.graph)

    sum = 0
    for i in range(0, x_test.shape[0]):
        if abs(y_test[i] - y_prediction[i]) < eps:
            sum += 1
    print("sum:", sum, "\n===================")

    # print detail
    # for i in np.random.randint(0,1000,500):
    #     print("i: ",i)
    #     print("x_test: ",x_test[i])
    #     print("y_test,outputs:")
    #     print(y_test[i],y_get[i])

    accuracy = tf.reduce_mean(tf.cast(abs(tf.add(y_sample, -outputs)) < eps, tf.float32))
    print(accuracy.eval(feed_dict={y_sample: y_test, x_sample: x_test}))

def sort_data(x_data, y_data, length):

    tmp = [(x_data[i], y_data[i]) for i in range(length)]
    tmp.sort(key=lambda o: o[0])
    tmp = list(map(list, zip(*tmp)))  # row-column transformation
    return tmp

if __name__ == "__main__":

    with tf.name_scope("inputs"):
        x_sample = tf.placeholder(tf.float32, [None, 1])
        y_sample = tf.placeholder(tf.float32, [None, 1])

    # Tensorflow 1.1.0
    # hide_layer1 = add_layers(1, x_sample, 1, 100, activation_function=None)
    # hide_layer2 = add_layers(2, hide_layer1, 100, 100, activation_function=tf.nn.relu)
    # outputs = add_layers(3, hide_layer2, 100, 1, activation_function=None)

    # Tensorflow 1.8.0
    hide_layer1 = tf.layers.dense(x_sample, 100, tf.nn.tanh)
    hide_layer2 = tf.layers.dense(hide_layer1, 100)
    outputs = tf.layers.dense(hide_layer2, 1)

    with tf.name_scope("loss"):
        # Tensorflow 1.1.0
        # error_sum = tf.reduce_sum(tf.square(y_sample - outputs), reduction_indices=[1])
        # loss = tf.reduce_mean(error_sum)

        # Tensorflow 1.8.0
        loss = tf.losses.mean_squared_error(y_sample, outputs)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)

    with tf.name_scope("train"):
        train = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        fig = plt.figure(figsize=(10,5))
        ax_figure = fig.add_subplot(1, 2, 1)
        ax_loss = fig.add_subplot(1,2,2)
        plt.ion()   # turn on the interactive mode

        loss_list = []

        for i in range(4000):

            # x_data = np.linspace(-1, 1, batch_size)[:, np.newaxis] same to the next row
            x_data = np.random.normal(0, 1, [batch_size, 1])
            noise = np.random.normal(0, 0.05, x_data.shape)
            y_data = f(x_data) + noise

            sess.run(train, feed_dict={x_sample: x_data, y_sample: y_data})

            if i % 200 == 0:

                try:
                    ax_figure.lines.remove(lines[0])
                    ax_figure.lines.remove(scatters[0])
                    ax_loss.remove()
                except Exception:
                    pass

                tmp = sort_data(x_data,y_data, 50)
                scatters = ax_figure.scatter(tmp[0], tmp[1], color='y', marker='+')

                y_prediction = sess.run(outputs, feed_dict={x_sample: x_data})

                tmp = sort_data(x_data, y_data, 50)
                lines = ax_figure.plot(tmp[0], tmp[1], 'r-', lw=3)

                plt.pause(0.1)

                loss_nw = sess.run(loss, feed_dict={x_sample:x_data, y_sample : y_data})
                print(loss_nw)
                loss_list.append(loss_nw)

                ax_loss.plot(range(len(loss_list)),loss_list, c='b')

        plt.ioff()  # turn off the interactive mode
        plt.show()

        test_model()