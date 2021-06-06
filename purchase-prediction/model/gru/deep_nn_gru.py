import numpy as np
import tensorflow as tf

from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.vis_utils import plot_model, model_to_dot

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

        x_train = []
        x_test = []

        for i in range(8):
            x_train.append(X_train[:, :, i])
            x_test.append(X_test[:, :, i])

        X_train = []
        X_test = []
        for i in range(8):
            X_train.append(x_train[i])
            X_test.append(x_test[i])

        return X_train, X_test, y_train, y_test

    @staticmethod
    def get_input_and_embedding_layers(X_train_enc):
        in_layers = list()
        em_layers = list()

        # add layer for price
        in_layer_price = tf.keras.layers.Input(shape=(4, 1), name='price')
        in_layers.append(in_layer_price)
        em_layers.append(in_layer_price)

        # add layer for discount
        in_layer_discount = tf.keras.layers.Input(shape=(4, 1), name='discount')
        in_layers.append(in_layer_discount)
        em_layers.append(in_layer_discount)

        # in_layer = tf.keras.layers.Input(shape=(1, 4, 1))

        for j in range(2, len(X_train_enc)):
            if j == 4:  # month
                name = 'month'
                n_labels = 12
            elif j == 5:  # day of the month
                name = 'day'
                n_labels = 31
            elif j == 6:  # day of the week
                name = 'week_day'
                n_labels = 7
            elif j == 7:  # hour
                name = 'hour'
                n_labels = 24
            else:
                # calculate the number of unique inputs
                n_labels = len(np.unique(X_train_enc[j]))

            if j == 2:
                name = 'category_path'
            elif j == 3:
                name = 'city'
            # define input layer
            in_layer = tf.keras.layers.Input(shape=(4,), name=name)
            # define embedding layer
            em_layer = tf.keras.layers.Embedding(n_labels, 10, input_length=4)(in_layer)
            # store layers
            in_layers.append(in_layer)
            em_layers.append(em_layer)

        return in_layers, em_layers

    @staticmethod
    def build(in_layers, em_layers):
        merge = tf.keras.layers.concatenate(axis=-1, inputs=em_layers)
        mask = tf.keras.layers.Masking(mask_value=0)(merge)
        gru = tf.compat.v1.keras.layers.GRU(248,
                                            recurrent_regularizer=tf.keras.regularizers.L2(0.01),
                                            activity_regularizer=tf.keras.regularizers.L2(0.01)
                                            )(mask)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(gru)
        model = tf.keras.Model(inputs=in_layers, outputs=output)

        plot_model(model, to_file='output/jpg/gru_model.jpg', show_shapes=False)
        # compile the keras model
        opt = keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-07)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        print(model.summary())

        return model

    @staticmethod
    def fit(model, X_train, X_test, y_train, y_test):
        # fit the keras model on the dataset
        model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=1)

        # evaluate the keras model
        _, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print('Test set accuracy: %.2f' % (accuracy * 100))

    @staticmethod
    def test_predict(model):
        predict_data = list()

        data = np.array([58.97, 1, 14, 6, 0, 23, 6, 16])
        data = data.reshape(1, 8)
        for i in range(8):
            predict_data.append(data[:, i])
        print("\nPrediction for data: [58.97, 1, 14, 6, 0, 23, 6, 16]\n")
        print(model.predict(predict_data))
