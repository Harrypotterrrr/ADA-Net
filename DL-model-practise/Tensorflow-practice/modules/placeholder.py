import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1,input2)

with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())  this is unnecessary
    prt = sess.run(output, feed_dict={input1:5.2,input2:3.0})
    print(prt)