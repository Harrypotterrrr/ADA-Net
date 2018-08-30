# coding: utf-8
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
import keras.callbacks
import matplotlib.pyplot as plt
from keras.utils import plot_model

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

######################################
# TODO: set the gpu memory using fraction
#####################################
def get_session(gpu_fraction=0.9):
    #to allocate GPU memory a specific fraction
    # Assume that you have 6GB of GPU memory and want to allocate ~2GB

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    input("Press Enter to continue...")

# Draw function
def training_vis(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']

    # make a figure
    fig = plt.figure(figsize=(8,4))

    # subplot loss
    ax1 = fig.add_subplot(121)
    ax1.plot(loss,label='train_loss')
    ax1.plot(val_loss,label='val_loss')
    ax1.scatter(history.epoch, history.history['loss'], marker='*')
    ax1.scatter(history.epoch, history.history['val_loss'], marker='*')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Training and Validation Data')
    ax1.legend(loc='upper right')

    # subplot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(acc,label='train_acc')
    ax2.plot(val_acc,label='val_acc')
    ax2.scatter(history.epoch, history.history['acc'], marker='*')
    ax2.scatter(history.epoch, history.history['val_acc'])
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy  on Training and Validation Data')
    ax1.legend(loc='under right')

    # plt.subplot(121)
    # plt.plot(history.epoch, history.history['acc'], label="acc")
    # plt.plot(history.epoch, history.history['val_acc'], label="val_acc")
    # plt.scatter(history.epoch, history.history['acc'], marker='*')
    # plt.scatter(history.epoch, history.history['val_acc'])
    # plt.legend(loc='under right')
    # plt.subplot(122)
    # plt.plot(history.epoch, history.history['loss'], label="loss")
    # plt.plot(history.epoch, history.history['val_loss'], label="val_loss")
    # plt.scatter(history.epoch, history.history['loss'], marker='*')
    # plt.scatter(history.epoch, history.history['val_loss'], marker='*')
    # plt.legend(loc='upper right')

    # Automatically adjust subplot parameters to give specified padding.
    plt.tight_layout()

if __name__ == "__main__":

    batch_size = 128
    nb_classes = 10
    nb_epoch = 10
    nb_data = 28 * 28
    path = r"./NN_test"

    # set the utilization of GPU / CPU
    '''
    # using 90% of total GPU Memory
    KTF.set_session(get_session(0.9))
    # Execute the command (a string) in a subshell
    os.system("nvidia-smi")
    '''

    # load data
    file = np.load(r"mnist.npz")
    (X_train, y_train) = file['x_train'],file['y_train']
    (X_test, y_test) = file['x_test'],file['y_test']

    # reshape
    print(X_train.shape)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

    # rescale
    X_train = X_train.astype('float32')
    # X_train = X_train.astype(np.float32)
    X_train /= 255

    X_test = X_test.astype(np.float32)
    X_test /= 255

    # convert class vectors to binary class matrices (one hot vectors)
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    # build the network
    model = Sequential()
    model.add(Dense(64, input_shape=(nb_data,), init='normal', name='dense1'))
    # a sample is row in 28 * 28
    model.add(Activation('relu', name='relu1'))
    model.add(Dropout(0.2, name='dropout1'))
    model.add(Dense(128, init='normal', name='dense2'))
    model.add(Activation('relu', name='relu2'))
    model.add(Dropout(0.2, name='dropout2'))
    model.add(Dense(10, init='normal', name='dense3'))
    model.add(Activation('softmax', name='softmax1'))

    # view model's parameters
    model.summary()

    # optimizer
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)

    # model compile for initializing the training parameter
    model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics=['accuracy'])

    # TensorBoard BLOCK
    # Set the storage location of the log, keep the network weights in the image format in the tensorboard,
    # set the weights of the network which calcuate once per cycle
    # distribute the histogram of the output values of each layer.
    tb_cb = keras.callbacks.TensorBoard(log_dir=path, write_images=1, histogram_freq=1)

    # change type to LIST
    cbks = [tb_cb]

    #load history weight value which has been trained and saved
    # model.load_weights(filepath = path + r'/NN_test_weight')

    #train the network
    history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                        verbose=1, callbacks=cbks, validation_data=(X_test, Y_test))

    # save training weight parameter
    model.save_weights(filepath = path+ r'/NN_test_weight2')

    # evaluation
    score = model.evaluate(X_test, Y_test, verbose=1)


    print('Test score:', score[0])
    print('Test accuracy;', score[1])

    # use keras graphvisulization to view the structure of network
    plot_model(model, show_shapes=True, to_file = path + r'/keras-cnn.png')

    # Draw
    training_vis(history)
    plt.show()