#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import numpy as np
import tensorflow as tf

"""
Custom Callback class intended for data collection over the course of training
"""
class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, training_data = None, test_data = None):
        super(CustomModelCheckpoint, self).__init__()
        self.model = None
        self.X_train = training_data
        self.X_test = test_data
        self.Y_train = []
        self.Y_test = []
        
    def get_train_data(self, epoch):
        return self.Y_train[epoch]

    def get_test_data(self, epoch):
        return self.Y_test[epoch]
    
    def on_epoch_end(self, epoch, logs=None):
        # if epoch % self.epoch_step == self.epoch_step-1:
        self.Y_train.append(self.model(self.X_train).numpy())
        self.Y_test.append(self.model(self.X_test).numpy())
"""
Available target functions to try and train against
"""
def function_sinusoid(X):
    return 5.0 * np.sin(2.0 * np.pi / 5.0 * X)

def function_quadratic(X):
    return 2.0 * np.power(X, 2.0) - 5.0 * X

def function_exponential(X):
    return X * np.exp( - np.power(X / 1.5, 2.0))

def function_piecewise(X):
    Xpiece1 = np.abs(X) >= 0.5
    Xpiece2 = np.abs(X) < 0.5
    Xpiece3 = np.abs(X) > 3
    return -0.5 * np.sign(X) * Xpiece1 + -X * Xpiece2 + (X - 3 * np.sign(X)) * Xpiece3

def function_segmentation(X):
    Xon = np.bitwise_or(np.bitwise_and(X >= -3, X < -1),
                        np.bitwise_and(X >= 0, X < 2))
    return -1.0 * ~Xon + 1.0 * Xon

"""
Available 'normalization' functions for compressing data
"""
def minmax_normalize(X):
    Xmax = np.max(X)
    Xmin = np.min(X)
    Xscale = 1./(Xmax - Xmin)
    Xshift = 0.5*(Xmax + Xmin)
    return Xscale, Xshift

def unity_normalize(X):
    return 1.0, 0.0

def get_model(output_scale, output_shift):
    # Neural network as a feed-forward neural network
    # Creates input layer as having single input
    inputs = tf.keras.layers.Input(shape=(1,))
    # Creates hidden Dense layer with ReLU activations at each node
    hidden1 = tf.keras.layers.Dense(16, activation=tf.keras.activations.relu)(inputs)
    # Creates hidden Dense layer with ReLU activations at each node
    hidden2 = tf.keras.layers.Dense(16, activation=tf.keras.activations.tanh)(hidden1)
    # Creates output Dense layer with linear activation
    outputs = tf.keras.layers.Dense(1)(hidden2)
    outputs = output_scale * outputs + output_shift
    model = tf.keras.models.Model(inputs, outputs)
    return model

def display_results(training_history, checkpoint_callback, Xtrain, Ytrain, Xtest, Ytest):
    Ymax = np.max(Ytest)
    Ymin = np.min(Ytest)
    Yrng = Ymax - Ymin

    Xmax = np.max(Xtest)
    Xmin = np.min(Xtest)
    Xrng = Xmax - Xmin

    # grab test and training data from collected data
    Ynn = checkpoint_callback.get_train_data(-1)
    Ynn_ext = checkpoint_callback.get_test_data(-1)
    
    _, [ax1, ax2] = plt.subplots(2, 1)
    # show the loss as a function of training epochs
    ax1.semilogy(training_history.epoch, training_history.history['loss'])
    ax1.set(xlabel='epoch', ylabel='loss')
    ax1.grid()

    # show the data plotted with ground truth behind
    ax2.plot(Xtest, Ytest, color='black')
    h_ynn_l, = ax2.plot(Xtest, Ynn_ext, '--', color='black', alpha=0.5)
    ax2.plot(Xtrain, Ytrain, 'o', color='green')
    h_ynn_pts, = ax2.plot(Xtrain, Ynn, 'x', color='blue')
    ax2.axis([Xmin, Xmax, Ymin - 0.25*Yrng, Ymax + 0.25*Yrng])
    ax2.set(xlabel='X', ylabel='Y')
    ax2.grid()

    # create indicater bar on the loss axes to move along with slider
    loss_min = np.min(training_history.history['loss'])
    loss_max = np.max(training_history.history['loss'])
    h_l, = ax1.semilogy(
        [training_history.epoch[-1], training_history.epoch[-1]],
        [loss_min, loss_max],
        color='red'
    )

    # event-called function to update figure according to slider position
    def update_slider(val): 
        # move indicater bar on loss axes to selected epoch
        h_l.set_xdata([training_history.epoch[val], training_history.epoch[val]])
        # get collected train/test data from model checkpoint callback history
        #   and set as display data
        Ynn_ext = checkpoint_callback.get_test_data(val)
        Ynn = checkpoint_callback.get_train_data(val)
        h_ynn_l.set_ydata(Ynn_ext)
        h_ynn_pts.set_ydata(Ynn)

    # allocate axes area for the slider bar under both plots
    axepoch = plt.axes([0.25, 0.03, 0.5, 0.02])
    # Create Slider for letting user select epoch to display
    sepoch = Slider(
        axepoch, 
        'epoch', 
        training_history.epoch[0], 
        training_history.epoch[-1], 
        valinit=training_history.epoch[-1], 
        valstep=1
    )
    # Register update_slider as response to 'on_changed' event
    sepoch.on_changed(update_slider)

    plt.show()

def main(function_name, max_epochs=1000, batch_size=16):
    # Generate dataset according to provided function
    available_functions = {
        'sinusoid': function_sinusoid, 
        'quadratic': function_quadratic, 
        'exponential': function_exponential, 
        'piecewise': function_piecewise, 
        'segmentation': function_segmentation
    }

    if not type(function_name) is str:
        print("function_name must be a string")
        return
    elif not function_name in available_functions:
        print("{} not an available function".format(function_name))
        for f in available_functions.keys():
            print("-f {}".format(f))
        return

    Fx = available_functions[function_name]

    # Random uniform sampling within X range
    N = 256
    Xrng = 6.0
    X = Xrng * 2.0 * (np.random.rand(N, 1) - 0.5)
    Y = Fx(X)

    # normalize data to [-1, 1]
    Xscale, Xshift = minmax_normalize(X)
    Xnorm = Xscale*(X - Xshift)

    Yscale, Yshift = minmax_normalize(Y)
    # Ynorm = Yscale*(Y - Yshift)

    # test results
    Xext = np.arange(-2.0 * Xrng, 2.0 * Xrng, 0.01)
    Xnorm_ext = Xscale*(Xext - Xshift)
    Yext = Fx(Xext)

    # Generate model
    model = get_model(1/Yscale, Yshift)

    # Compile model with optimizer, loss function, and captured metrics for optimization during
    #   backpropogation
    #   Adam defaults to learning_rate = 0.001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.01),
                    loss=tf.keras.losses.MeanSquaredError(),
                    metrics=[]
                    )
    
    custom_model_checkpoint = CustomModelCheckpoint(training_data = Xnorm, test_data = Xnorm_ext)

    # Actual backpropagation
    #   batch_size defaults to 32 if unspecified / set to None
    #   epochs defaults to 1 if unspecified
    training_history = model.fit(
        x=Xnorm, 
        y=Y, 
        batch_size=batch_size, 
        callbacks=[custom_model_checkpoint],
        epochs=max_epochs
    )
    # # Equivalent to model.fit (very slow relative to optimized model.fit, but equivalent performance)
    # Ntrain = np.size(Y, 0)
    # samples_per_epoch = Ntrain - (Ntrain % batch_size)
    # batches_per_epoch = int(samples_per_epoch / batch_size)
    # training_history = tf.keras.callbacks.History()
    # training_history.set_model(model)
    # custom_model_checkpoint.set_model(model)

    # training_history.on_train_begin()
    # for epoch in range(0, max_epochs):
    #     # sample minibatches from training dataset
    #     sample_idxs = np.random.permutation(Ntrain)[:samples_per_epoch]
    #     batch_idxs = sample_idxs.reshape((batches_per_epoch, batch_size))
    #     custom_model_checkpoint.on_epoch_begin(epoch)
    #     training_history.on_epoch_begin(epoch)
    #     epoch_logs = dict()
    #     epoch_logs['loss'] = 0
    #     for batch in batch_idxs:
    #         x_batch = Xnorm[batch]
    #         y_batch = Y[batch]
    #         with tf.GradientTape() as tape:
    #             logits = model(x_batch)
    #             loss_value = model.loss(y_batch, logits)
            
    #         model_gradient = tape.gradient(loss_value, model.trainable_variables)
    #         model.optimizer.apply_gradients(zip(model_gradient, model.trainable_variables))

    #         epoch_logs['loss'] += loss_value

    #     print("epoch {} : loss: {}".format(epoch, loss_value))
    #     custom_model_checkpoint.on_epoch_end(epoch)
    #     training_history.on_epoch_end(epoch, epoch_logs)
    


    # display results
    display_results(training_history, custom_model_checkpoint, X, Y, Xext, Yext)
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = '1D Tensorflow example')
    parser.add_argument('-f', '--function', help='function name')
    parser.add_argument('-e', '--epochs', type=int, help='epochs to train over')
    parser.add_argument('-b', '--batch', type=int, help='batch size')
    args = parser.parse_args()
    main(args.function, batch_size=args.batch, max_epochs=args.epochs)