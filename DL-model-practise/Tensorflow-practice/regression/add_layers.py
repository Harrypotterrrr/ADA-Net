import tensorflow as tf

def add_layers(k, inputs, in_size, out_size, activation_function=None):

    layer_name = "layer %d" % k
    with tf.name_scope("layer"):
        with tf.name_scope("Weights"):
            Weights = tf.Variable(tf.random_normal([in_size, out_size],mean=0,stddev=0.1), name = "w")
            # tf.histogram_summary(layer_name+'/weights', Weights)

        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name="b")
            # tf.histogram_summary(layer_name + '/biases', biases)

        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)

        # tf.histogram_summary(layer_name+'/outputs', outputs)
        return outputs
