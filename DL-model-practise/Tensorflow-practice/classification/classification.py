import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TP_CPP_MIN_LOG_LEVEL']='2'

data_size = 200
data_mean = 2
data_stddev = 1.5
dropout_rate = 0.5
training_loop = 4000
print_cycle = 10
learning_rate_init = 0.05

decay_rate = 0.94
decay_steps = 100

def create_color():
    y_color = [0.1] * data_size
    y_color.extend([0.4] * data_size)
    y_color.extend([0.7] * data_size)
    y_color.extend([0.9] * data_size)
    return y_color

def create_data():
    x0 = [np.random.normal([data_mean, data_mean], data_stddev) for i in range(data_size)]
    x1 = [np.random.normal([-data_mean, data_mean], data_stddev) for i in range(data_size)]
    x2 = [np.random.normal([-data_mean, -data_mean], data_stddev) for i in range(data_size)]
    x3 = [np.random.normal([data_mean, -data_mean], data_stddev) for i in range(data_size)]
    x = np.vstack((x0, x1, x2, x3))
    y0 = [[1, 0, 0, 0] for i in range(data_size)]
    y1 = [[0, 1, 0, 0] for i in range(data_size)]
    y2 = [[0, 0, 1, 0] for i in range(data_size)]
    y3 = [[0, 0, 0, 1] for i in range(data_size)]
    y = np.vstack((y0, y1, y2, y3))
    return x, y

if __name__ == "__main__":
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    x, y = create_data()
    y_color = create_color()
    ax1.scatter(x[:, 0], x[:, 1], c=y_color, cmap="rainbow")

    with tf.name_scope("input"):
        x_sample = tf.placeholder(tf.float32, x.shape)
        y_sample = tf.placeholder(tf.float32, y.shape)

    layer1 = tf.layers.dense(x_sample, 100, activation=tf.nn.tanh)
    layer2 = tf.layers.dense(layer1, 100, activation=tf.nn.tanh)
    layer_drop = tf.layers.dropout(layer2, rate=dropout_rate)
    outputs = tf.layers.dense(layer_drop, 4)

    # NOTE："tf.losses.sparse_softmax_cross_entropy()" demand the labelis a scalar("sparse"),
    # instead of 4-dimension on-hot vector, link: https://blog.csdn.net/wenzishou/article/details/75810078
    # "tf.losses.sparse_softmax_cross_entropy()" is the updated version of:
    # "tf.nn.sparse_softmax_cross_entropy_with_logits()"
    with tf.name_scope("loss"):
        # version 1.
        loss_sigmoid = tf.losses.sigmoid_cross_entropy(multi_class_labels=y_sample, logits=outputs)

        # version 2.
        loss_softmax = tf.losses.softmax_cross_entropy(onehot_labels=y_sample, logits=outputs)

        # version 3.
        # error_sum = tf.reduce_sum(tf.add(y_sample, -outputs))
        # loss = -tf.reduce_sum(y_sample * tf.log(outputs))

        # version 4.
        # loss_sigmoid = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_sample, logits=outputs)

        # version 5.
        # loss_softmax = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_sample, logits=outputs)

        # WRONG-version 6.
        # loss_sigmoid = tf.losses.sparse_sigmoid_cross_entropy(labels=y_sample, logits=outputs)


    with tf.name_scope("learning_rate"):
        steps = tf.Variable(tf.constant(0))
        lr_decay = tf.train.exponential_decay(
            learning_rate_init,
            steps,
            decay_steps,
            decay_rate,
            staircase = True
        )

    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
        train_sigmoid = optimizer.minimize(loss_sigmoid)
        train_softmax = optimizer.minimize(loss_softmax)

    with tf.name_scope("accuracy"):
        error_bool = tf.equal(tf.argmax(y_sample, axis=1), tf.argmax(outputs, axis=1))
        error = tf.cast(error_bool, tf.float32)
        accuracy = tf.reduce_mean(error)

    # with tf.Session() as debug:
    #     init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    #     debug.run(init)  # initialize var in graph
    #     l,eb,e,ot,acc= debug.run(
    #         [loss, error_bool, error, outputs, accuracy],
    #         feed_dict={
    #             x_sample: x,
    #             y_sample: y
    #         }
    #     )
    #     print(eb)
    #     print(e)
    #     print(ot)
    #     print(acc)
    #     input()
    # ####################################

    with tf.Session() as sess:

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)  # initialize var in graph


        loss_list = []
        acc_list = []
        for i in range(training_loop):

            _, lr, loss_, acc_, outputs_ = sess.run(
                [train_sigmoid, lr_decay, loss_sigmoid, accuracy, outputs],
                feed_dict={
                    x_sample: x,
                    y_sample: y,
                    steps: i
                }
            )

            if i % print_cycle == 0:
                acc_list.append(acc_)
                # print("accuracy: ", acc_)
                # print("loss: ", loss_)

        ax2.plot(range(0,training_loop,print_cycle), acc_list, color='orange', label="sigmoid")

    print("===================================")

    with tf.Session() as sess:

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)  # initialize var in graph

        loss_list = []
        acc_list = []
        for i in range(training_loop):

            _, lr, loss_, acc_, outputs_ = sess.run(
                [train_softmax, lr_decay, loss_softmax, accuracy, outputs],
                feed_dict={
                    x_sample: x,
                    y_sample: y,
                    steps: i
                }
            )

            if i % print_cycle == 0:
                acc_list.append(acc_)
                # print("accuracy: ", acc_)
                # print("loss: ", loss_)

        ax2.plot(range(0,training_loop,print_cycle), acc_list, color='red', label='softmax')

    plt.legend(loc="lower right")
    plt.show()
