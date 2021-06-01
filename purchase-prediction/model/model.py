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

from model.prepare_dataset import load_dataset, prepare_inputs, prepare_targets, load_sequential_dataset, reshape_sequential_inputs


def get_dataset():
    # load the dataset
    X, y = load_dataset('output/', 'sessions.csv')
    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=1)
    # prepare input data
    X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
    # prepare output data
    y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
    # make output 3d
    y_train_enc = y_train_enc.reshape((len(y_train_enc), 1, 1))
    y_test_enc = y_test_enc.reshape((len(y_test_enc), 1, 1))

    return X_train_enc, X_test_enc, y_train_enc, y_test_enc


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


def get_input_and_embedding_layers(X_train_enc):
    # prepare each input head
    in_layers = list()
    em_layers = list()

    # add layer for price
    in_layer_price = tf.keras.layers.Input(shape=(1, 1))
    in_layers.append(in_layer_price)
    em_layers.append(in_layer_price)

    for i in range(1, len(X_train_enc)):
        # calculate the number of unique inputs
        n_labels = len(unique(X_train_enc[i]))
        # define input layer
        in_layer = tf.keras.layers.Input(shape=(1,))
        # define embedding layer
        em_layer = tf.keras.layers.Embedding(n_labels, 10)(in_layer)
        # store layers
        in_layers.append(in_layer)
        em_layers.append(em_layer)

    return in_layers, em_layers


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


def build_model(em_layers, in_layers):
    # concat all layers(price + embeddings)
    merge = tf.keras.layers.concatenate(axis=-1, inputs=em_layers)
    dense = tf.keras.layers.Dense(10, activation='relu', kernel_initializer='he_normal')(merge)
    dense = tf.keras.layers.Dense(5, activation='relu', kernel_initializer='he_normal')(dense)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dense)
    model = tf.keras.Model(inputs=in_layers, outputs=output)

    # compile the keras model
    opt = keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-07)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def build_sequence_model(in_layers, em_layers):
    # concat all layers (price + embeddings)
    merge = tf.keras.layers.concatenate(axis=-1, inputs=em_layers)
    """
        Both GRU and LSTM throw __len__ error
    """
    gru = tf.keras.layers.GRU(100, return_sequences=True)(merge)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(gru)
    model = tf.keras.Model(inputs=in_layers, outputs=output)

    # compile the keras sequence model
    opt = keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-07)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    # plot graph
    # plot_model(model, to_file='output/sequence_model.png', show_shapes=True)

    return model


def build_hyper_model(hp):
    # Input : hyperparameter tuner
    # Output : Model object

    # get train and test sets
    X_train_enc, X_test_enc, y_train_enc, y_test_enc = get_dataset()
    # generate input and embedding layers
    in_layers, em_layers = get_input_and_embedding_layers(X_train_enc)
    # initialize functional model
    merge = tf.keras.layers.concatenate(axis=-1, inputs=em_layers)
    dense = tf.keras.layers.Dense(10, activation='relu', kernel_initializer='he_normal')(merge)

    # for loop for hidden layers
    for i in range(hp.Int('num_layers', min_value=1, max_value=2, step=1)):
        # add hidden layer and
        # finding units for each layer
        dense = (tf.keras.layers.Dense(units=hp.Int('units' + str(i),
                                                    min_value=4,
                                                    max_value=5,
                                                    step=1),
                                       activation=hp.Choice('activation_function' + str(i),
                                                            values=['linear', 'relu', 'sigmoid', 'tanh']),
                                       kernel_initializer='he_normal'))(dense)

    output = tf.keras.layers.Dense(1, activation='sigmoid')(dense)
    model = tf.keras.Model(inputs=in_layers, outputs=output)

    # add optimizers and metrics
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # plot_model(model, to_file='model/hypertuning/hyper_model.png', show_shapes=True)

    return model


def fit_model(X_train_enc, X_test_enc, y_train_enc, y_test_enc, em_layers, in_layers):
    # define model
    model = build_model(em_layers, in_layers)

    # fit the keras model on the dataset
    model.fit(X_train_enc, y_train_enc, epochs=300, batch_size=128, verbose=2)
    # evaluate the keras model
    _, accuracy = model.evaluate(X_test_enc, y_test_enc, verbose=0)
    print('Accuracy: %.2f' % (accuracy * 100))

    # plot graph
    # plot_model(model, to_file='output/model.png', show_shapes=True)
