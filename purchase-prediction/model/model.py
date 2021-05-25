"""
    Name: model.py
    Purpose:

    @author Bartosz Świtalski, Piotr Frątczak

    Warsaw University of Technology
    Faculty of Electronics and Information Technology
"""
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import concatenate, Dense, Embedding, Input
from tensorflow.keras.models import Model
from keras.utils.vis_utils import plot_model
from preprocess.prepare_dataset import load_dataset, prepare_inputs, prepare_targets

from numpy import unique


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


def get_input_and_embedding_layers(X_train_enc):
    # prepare each input head
    in_layers = list()
    em_layers = list()

    # add layer for price
    in_layer_price = Input(shape=(1, 1))
    in_layers.append(in_layer_price)
    em_layers.append(in_layer_price)

    for i in range(1, len(X_train_enc)):
        # calculate the number of unique inputs
        n_labels = len(unique(X_train_enc[i]))
        # define input layer
        in_layer = Input(shape=(1,))
        # define embedding layer
        em_layer = Embedding(n_labels, 10)(in_layer)
        # store layers
        in_layers.append(in_layer)
        em_layers.append(em_layer)

    return in_layers, em_layers


def build_model(em_layers, in_layers):
    # concat all layers(price + embeddings)
    merge = concatenate(axis=-1, inputs=em_layers)
    dense = Dense(10, activation='relu', kernel_initializer='he_normal')(merge)
    dense = Dense(4, activation='relu', kernel_initializer='he_normal')(dense)
    output = Dense(1, activation='sigmoid')(dense)
    model = Model(inputs=in_layers, outputs=output)

    # compile the keras model
    opt = keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-07)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def build_hyper_model(hp):
    # Input : hyperparameter tuner
    # Output : Model object

    # get train and test sets
    X_train_enc, X_test_enc, y_train_enc, y_test_enc = get_dataset()
    # generate input and embedding layers
    in_layers, em_layers = get_input_and_embedding_layers(X_train_enc)
    # initialize functional model
    merge = concatenate(axis=-1, inputs=em_layers)
    dense = Dense(10, activation='relu', kernel_initializer='he_normal')(merge)

    # for loop for hidden layers
    for i in range(hp.Int('num_layers', min_value=1, max_value=3, step=1)):
        # add hidden layer and
        # finding units for each layer
        dense = (Dense(units=hp.Int('units' + str(i),
                                    min_value=4,
                                    max_value=6,
                                    step=1)))(dense)

    output = Dense(1, activation='sigmoid')(dense)
    model = Model(inputs=in_layers, outputs=output)

    # add optimizers and metrics
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    plot_model(model, to_file='model/hypertuning/hyper_model.png', show_shapes=True)

    return model


def fit_model(X_train_enc, X_test_enc, y_train_enc, y_test_enc, em_layers, in_layers):
    # define model
    model = build_model(em_layers, in_layers)

    # plot graph
    plot_model(model, to_file='output/embeddings.png', show_shapes=True)
    # fit the keras model on the dataset
    model.fit(X_train_enc, y_train_enc, epochs=300, batch_size=128, verbose=2)
    # evaluate the keras model
    _, accuracy = model.evaluate(X_test_enc, y_test_enc, verbose=0)
    print('Accuracy: %.2f' % (accuracy * 100))
