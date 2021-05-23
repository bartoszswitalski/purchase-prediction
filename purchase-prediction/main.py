"""
    Name:
    Purpose:

    @author Bartosz Świtalski, Piotr Frątczak

    Warsaw University of Technology
    Faculty of Electronics and Information Technology
"""
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from numpy import unique
from sklearn.model_selection import train_test_split

from preprocess.aggregate import get_aggregated_sessions
from preprocess.prepare_dataset import load_dataset, prepare_inputs, prepare_targets
from keras.layers import Input, concatenate, Dense
from keras.layers import Embedding
from keras.models import Model

if __name__ == '__main__':

    # limit GPU usage
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # don't pre-allocate memory; allocate as-needed
    config.gpu_options.per_process_gpu_memory_fraction = 0.95  # limit memory to be allocated
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))  # create sess w/ above settings

    # load the dataset
    X, y = load_dataset('output/', 'sessions.csv')
    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    # prepare input data
    X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
    # prepare output data
    y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
    # make output 3d
    y_train_enc = y_train_enc.reshape((len(y_train_enc), 1, 1))
    y_test_enc = y_test_enc.reshape((len(y_test_enc), 1, 1))

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

    # concat all layers(price + embeddings)
    merge = concatenate(axis=-1, inputs=em_layers)
    dense = Dense(10, activation='relu', kernel_initializer='he_normal')(merge)
    output = Dense(1, activation='sigmoid')(dense)
    model = Model(inputs=in_layers, outputs=output)

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # plot graph
    plot_model(model, to_file='output/embeddings.png', show_shapes=True)
    # fit the keras model on the dataset
    model.fit(X_train_enc, y_train_enc, epochs=100, batch_size=128, verbose=2)
    # evaluate the keras model
    _, accuracy = model.evaluate(X_test_enc, y_test_enc, verbose=0)
    print('Accuracy: %.2f' % (accuracy * 100))
