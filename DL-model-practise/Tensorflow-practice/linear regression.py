import tensorflow as tf
import numpy as np
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='3' #Disable Tensorflow debugging information
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #Set GPU device -1 -> set cpu to operate

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

Weight = tf.Variable(tf.random_uniform([1],-1.0,1.0))
bias = tf.Variable(tf.zeros([1]))

y_predict = Weight * x_data + bias
loss = tf.reduce_mean(tf.square(y_predict - y_data))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2000):
    sess.run(train)
    print(step, sess.run(Weight), sess.run(bias))

