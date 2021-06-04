import numpy as np
import tensorflow as tf
import kerastuner as kt

from tensorflow import keras
from sklearn.model_selection import train_test_split

from model.simple.utils import load_dataset, prepare_inputs, prepare_targets


class DeepModel:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset():
        """
        Loads dataset from the file. Splits it into train and test inputs and targets.
        Encodes proper attributes. Reshapes target.

        :return: train inputs, test inputs, train targets, test targets
        """
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

    @staticmethod
    def get_input_and_embedding_layers(X_train_enc):
        """
        Creates input and embedding layers according to given train inputs.

        :param X_train_enc: train inputs
        :return: input layers, embedding layers
        """
        # prepare each input head
        in_layers = list()
        em_layers = list()

        # add layer for price
        in_layer_price = tf.keras.layers.Input(shape=(1, 1))
        in_layers.append(in_layer_price)
        em_layers.append(in_layer_price)

        for i in range(1, len(X_train_enc)):
            # calculate the number of unique inputs
            n_labels = len(np.unique(X_train_enc[i]))
            # define input layer
            in_layer = tf.keras.layers.Input(shape=(1,))
            # define embedding layer
            em_layer = tf.keras.layers.Embedding(n_labels, 10)(in_layer)
            # store layers
            in_layers.append(in_layer)
            em_layers.append(em_layer)

        return in_layers, em_layers

    @staticmethod
    def build(in_layers, em_layers):
        """
        Builds tf.keras.Model according to given input and embedding layers.
        The model has a concatenate layer for embeddings, two Dense layers with 10 and neurons and
        finally a Dense layer with single neuron and sigmoid activation.

        ADAM is used as the optimizer. Learning rate is set to 0.001 and epsilon is set to 1e-07

        Binary crossentropy is used as the loss function.

        :param in_layers: input layers
        :param em_layers: embedding layers
        :return: built tf.keras.Model
        """
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

    @staticmethod
    def build_hyper_model(hp):
        """

        :param hp:
        :return:
        """
        # get train and test sets
        X_train_enc, X_test_enc, y_train_enc, y_test_enc = DeepModel.get_dataset()
        # generate input and embedding layers
        in_layers, em_layers = DeepModel.get_input_and_embedding_layers(X_train_enc)
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

    @staticmethod
    def fit(model, X_train_enc, X_test_enc, y_train_enc, y_test_enc):
        """
        Fits given tf.keras.Model to given train datasets.
        Epochs are set to 300, batch size is set to 128.

        After fitting the evaluation of the model is being performed
        on the given test datasets. Accuracy is calculated.

        :param model: tf.keras.Model
        :param X_train_enc: train inputs
        :param X_test_enc: test inputs
        :param y_train_enc: train targets
        :param y_test_enc: test targets
        :return: none
        """
        print('\n-----------\n')
        print(type(X_train_enc))
        print(type(y_train_enc))

        print(X_train_enc)
        print(y_train_enc)
        print('\n-----------\n')
        # fit the keras model on the dataset
        model.fit(X_train_enc, y_train_enc, epochs=300, batch_size=128, verbose=2)
        # evaluate the keras model
        _, accuracy = model.evaluate(X_test_enc, y_test_enc, verbose=0)
        print('Accuracy: %.2f' % (accuracy * 100))

        # plot graph
        # plot_model(model, to_file='output/model.png', show_shapes=True)

    @staticmethod
    def tune(X_train_enc, X_test_enc, y_train_enc, y_test_enc):
        """
        Tunes hyper parameters on given datasets.
        Prints info while tuning after the summary afterwards.

        :param X_train_enc: train inputs
        :param X_test_enc: test inputs
        :param y_train_enc: train targets
        :param y_test_enc: test targets
        :return: none
        """
        # define hyper model
        tuner = kt.Hyperband(
            DeepModel.build_hyper_model, objective="val_accuracy", max_epochs=50, hyperband_iterations=5, overwrite=True,
            project_name='keras_tuner_config_output'
        )

        print(tuner.search_space_summary())

        tuner.search(X_train_enc, y_train_enc,
                     epochs=50,
                     # callbacks=[[tf.keras.callbacks.EarlyStopping(monitor='val_loss',  patience=5)]],
                     validation_data=(X_test_enc, y_test_enc))

        best_model = tuner.get_best_models(1)[0]
        best_hyper_parameters = tuner.get_best_hyperparameters(1)[0]
        tuner.results_summary()

        print(best_model)
        print(best_hyper_parameters)
