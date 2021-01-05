#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def main():
    # Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    # data shape:   x_train  -  np.array(60000, 28, 28)
    #               y_train  -  np.array(60000, )
    #               x_test   -  np.array(10000, 28, 28)
    #               y_test   -  np.array(10000, )
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Neural network 'Sequential' as a feed-forward neural network
    model = tf.keras.models.Sequential()
    # flattens inputs into a 784 node layer
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    # Creates hidden Dense layer with ReLU activations at each node
    #   Uses Glorot uniform initializer (also known as Xavier uniform initialization)
    #   by default for coefficients
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    # Creates Dropout between hidden layer and output with rate frequency = 0.2
    # Randomly sets input layer to 0 and scales remaining nodes by 1/(1 - rate)
    model.add(tf.keras.layers.Dropout(0.2))
    # Creates output Dense layer with linear activation
    #   Uses Glorot uniform initializer (also known as Xavier uniform initialization)
    #   by default for coefficients
    model.add(tf.keras.layers.Dense(10))

    # initial predictions for first training data point 
    init_predictions_1 = model(x_train[:1]).numpy()

    # initial 'probabilities' as according to softmax
    #   Recommended to not use softmax as direct output as it does not have a stable loss calculation
    #   for all models
    init_predictions_softmax_1 = tf.nn.softmax(init_predictions_1).numpy()

    # Creates cross-entropy loss function, which accepts output logit vector from model and trains against
    #   provided 'true' index.  
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    init_predictions_loss_1 = loss_fn(y_train[:1], init_predictions_1).numpy()

    idx = 0
    print("*************************************")
    print("first data point loss: {}".format(init_predictions_loss_1))
    for pred, softmax in np.nditer([init_predictions_1, init_predictions_softmax_1]):
        res_string = ""
        if idx == y_train[:1]:
            res_string = " * class, cross-entropy {}".format(-np.log(init_predictions_softmax_1[0, idx]))
        print("{}: {}, {}{}".format(idx, pred, softmax, res_string))
        idx+=1

    print("*************************************")

    # compile model to use 'Adam' gradient descent algorithm (Adaptive Momentum),
    #   provided categorical cross-entropy loss function
    #   and return accuracy metric
    model.compile(
        optimizer='adam',
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    # perform backpropagation to minimize parameters to the training set
    #   in TensorFlow 2, batch size defaults to 32
    #       -> will perform steps per epoch = total samples / batch_size
    #   returns keras.callbacks.History object with useful attributes
    #       epoch - array of epoch indexes
    #       history - dictionary of data captured at each epoch
    #       params - dictionary of parameters used such as 'verbose', 'epochs', 'steps'
    training_history = model.fit(
        x=x_train, 
        y=y_train, 
        epochs=5
    )

    # evaluate the model on the testing dataset to see loss
    #   verbose: 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch
    evaluation_results = model.evaluate(x_test, y_test, verbose=2)
    print("*************************************")
    print("test data loss: {}, accuracy: {}".format(evaluation_results[0], evaluation_results[1]))
    print("*************************************")

    # final predictions for first training data point 
    final_predictions_1 = model(x_train[:1]).numpy()

    # final 'probabilities' as according to softmax
    final_predictions_softmax_1 = tf.nn.softmax(final_predictions_1).numpy()

    final_predictions_loss_1 = loss_fn(y_train[:1], final_predictions_1).numpy()

    idx = 0
    print("*************************************")
    print("first data point loss: {}".format(final_predictions_loss_1))
    for pred, softmax in np.nditer([final_predictions_1, final_predictions_softmax_1]):
        res_string = ""
        if idx == y_train[:1]:
            res_string = " * class, cross-entropy {}".format(-np.log(final_predictions_softmax_1[0, idx]))
        print("{}: {}, {}{}".format(idx, pred, softmax, res_string))
        idx+=1

    print("*************************************")

    fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True)
    ax1.plot(training_history.epoch, training_history.history['accuracy'])
    ax1.set(ylabel='accuracy')
    ax1.grid()

    ax2.plot(training_history.epoch, training_history.history['loss'])
    ax2.set(xlabel='epoch', ylabel='loss')
    ax2.grid()

    plt.show()

if __name__ == "__main__":
    main()