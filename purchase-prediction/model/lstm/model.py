"""
    Name: model.py
    Purpose:

    @author Bartosz Świtalski, Piotr Frątczak

    Warsaw University of Technology
    Faculty of Electronics and Information Technology
"""
import tensorflow as tf
from tensorflow import keras
from numpy import unique
from sklearn.model_selection import train_test_split

# from keras.utils.vis_utils import plot_model

from model.lstm.prepare_dataset import load_sequential_dataset, reshape_sequential_inputs


def get_sequential_dataset():
    # load the dataset
    X, y = load_sequential_dataset('output/', 'sessions_encoded.csv')

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, shuffle=False)

    # prepare data
    X_train, X_test = reshape_sequential_inputs(X_train, X_test)

    # make output 3d
    y_train = y_train.reshape((len(y_train), 1, 1))
    y_test = y_test.reshape((len(y_test), 1, 1))

    return X_train, X_test, y_train, y_test


def get_input_and_embedding_layers_for_sequential(X_train_enc, mask_zero=False):
    in_layers_sequential = list()
    em_layers_sequential = list()

    for i in range(11):
        # add layer for price
        in_layer_price = tf.keras.layers.Input(shape=(1, 1))
        in_layers_sequential.append(in_layer_price)
        em_layers_sequential.append(in_layer_price)

        for i in range(1, len(X_train_enc)):
            # calculate the number of unique inputs
            n_labels = len(unique(X_train_enc[i]))
            # define input layer
            in_layer = tf.keras.layers.Input(shape=(1,))
            # define embedding layer
            em_layer = tf.keras.layers.Embedding(n_labels, 10, mask_zero=mask_zero)(in_layer)
            # store layers
            in_layers_sequential.append(in_layer)
            em_layers_sequential.append(em_layer)

    return in_layers_sequential, em_layers_sequential
