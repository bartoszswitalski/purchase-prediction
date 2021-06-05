import numpy as np
import tensorflow as tf

from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.vis_utils import plot_model

from model.gru.utils import load_sequential_dataset, prepare_inputs_for_sequential


class GRUModel:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset():
        X, y = load_sequential_dataset('output/', 'sessions_encoded.csv')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

        # X_train, X_test = prepare_inputs_for_sequential(X_train, X_test)
        y_train = y_train.reshape((len(y_train), 1, 1))
        y_test = y_test.reshape((len(y_test), 1, 1))

        return X_train, X_test, y_train, y_test

    @staticmethod
    def get_input_and_embedding_layers(X_train_enc, mask_zero=False):
        in_layers = list()
        em_layers = list()

        # for _ in range(4):
        #     # add layer for price
        #     in_layer_price = tf.keras.layers.Input(shape=(1, 1))
        #     in_layers.append(in_layer_price)
        #     em_layers.append(in_layer_price)
        #
        #     # add layer for discount
        #     in_layer_discount = tf.keras.layers.Input(shape=(1, 1))
        #     in_layers.append(in_layer_discount)
        #     em_layers.append(in_layer_discount)
        #
        #     for j in range(2, len(X_train_enc)):
        #         if j == 4:  # month
        #             n_labels = 12
        #         elif j == 5:  # day of the month
        #             n_labels = 31
        #         elif j == 6:  # day of the week
        #             n_labels = 7
        #         elif j == 7:  # hour
        #             n_labels = 24
        #         else:
        #             # calculate the number of unique inputs
        #             n_labels = len(np.unique(X_train_enc[j]))
        #         # define input layer
        #         in_layer = tf.keras.layers.Input(shape=(1,))
        #         # define embedding layer
        #         em_layer = tf.keras.layers.Embedding(n_labels, 10, mask_zero=mask_zero)(in_layer)
        #         # store layers
        #         in_layers.append(in_layer)
        #         em_layers.append(em_layer)

        # add layer for price
        in_layer_price = tf.keras.layers.Input(shape=(1, 1, 4))
        # in_layers.append(in_layer_price)
        em_layers.append(in_layer_price)

        # add layer for discount
        in_layer_discount = tf.keras.layers.Input(shape=(1, 1, 4))
        # in_layers.append(in_layer_discount)
        em_layers.append(in_layer_discount)

        in_layer = tf.keras.layers.Input(shape=(1, 8, 4))

        for j in range(2, len(X_train_enc)):
            if j == 4:  # month
                n_labels = 12
            elif j == 5:  # day of the month
                n_labels = 31
            elif j == 6:  # day of the week
                n_labels = 7
            elif j == 7:  # hour
                n_labels = 24
            else:
                # calculate the number of unique inputs
                n_labels = len(np.unique(X_train_enc[j]))
            # define input layer
            # in_layer = tf.keras.layers.Input(shape=(1, 4))
            # define embedding layer
            em_layer = tf.keras.layers.Embedding(n_labels, 10, input_length=4)(in_layer)
            # store layers
            # in_layers.append(in_layer)
            em_layers.append(em_layer)

        return in_layer, em_layers

    @staticmethod
    def build(in_layers, em_layers):
        # concat all layers(price + embeddings)
        # merge = tf.keras.layers.concatenate(axis=1, inputs=em_layers)
        inputs = tf.keras.layers.Input(shape=(4, 8))
        mask = tf.keras.layers.Masking(mask_value=0)(inputs)
        dense = tf.compat.v1.keras.layers.GRU(248,
                                              recurrent_regularizer=tf.keras.regularizers.L2(0.01),
                                              activity_regularizer=tf.keras.regularizers.L2(0.01)
                                              )(mask)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(dense)
        model = tf.keras.Model(inputs=inputs, outputs=output)

        plot_model(model, to_file='output/jpg/gru_model.jpg', show_shapes=True)
        # compile the keras model
        opt = keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-07)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        print(model.summary())

        return model

    @staticmethod
    def fit(model, X_train, X_test, y_train, y_test):
        # fit the keras model on the dataset
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2)

        # evaluate the keras model
        _, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print('Test set accuracy: %.2f' % (accuracy * 100))
