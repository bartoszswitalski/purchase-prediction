"""
    Name: main.py
    Purpose:

    @author Bartosz Świtalski, Piotr Frątczak

    Warsaw University of Technology
    Faculty of Electronics and Information Technology
"""
from preprocess.check_users import users_check
from preprocess.check_products import products_check
from preprocess.check_sessions import sessions_check

from model.simple.deep_nn import DeepModel
from model.gru.deep_nn_gru import GRUModel
from model.gpu_setup.usage import limit_gpu_usage
from model.gru.aggregate import get_aggregated_sessions

import numpy as np

if __name__ == '__main__':
    """
        uncomment to check data and save to sessions.csv and sessions_encoded.csv
    """
    # users_check()
    # products_check()
    # sessions_check()

    """
        uncomment when running the model
    """
    # # limit GPU usage
    # limit_gpu_usage()

    """
        uncomment to fit basic model
    """
    # # get train and test sets
    # X_train_enc, X_test_enc, y_train_enc, y_test_enc = DeepModel.get_dataset()
    # # generate input and embedding layers
    # in_layers, em_layers = DeepModel.get_input_and_embedding_layers(X_train_enc)
    # # build model
    # model = DeepModel.build(in_layers, em_layers)
    # # fit model
    # DeepModel.fit(model, X_train_enc, X_test_enc, y_train_enc, y_test_enc)
    # # test prediction
    # DeepModel.test_predict(model)

    """
        uncomment to tune parameters 
    """
    # # get train and test sets
    # X_train_enc, X_test_enc, y_train_enc, y_test_enc = DeepModel.get_dataset()
    # # tune parameters
    # DeepModel.tune(X_train_enc, X_test_enc, y_train_enc, y_test_enc)

    # """
    #     uncomment to build sequence model
    # """
    X_train_enc, X_test_enc, y_train_enc, y_test_enc = DeepModel.get_dataset()
    # get train and test sets
    X_train, X_test, y_train, y_test = GRUModel.get_dataset()
    print('\n---------\n')
    print(type(X_train))
    print(X_train.shape)
    print(X_train)

    # k = 0
    # X_train = X_train.tolist()
    # for i in range(len(X_train)):
    #     X_train[i] = np.asarray(X_train[i])
    #     sequenced_attributes = list()
    #     for j in range(8):
    #         sequenced_attributes.append(X_train[i][:, j])
    #     sequenced_attributes = np.asarray(sequenced_attributes)
    #     X_train[i] = sequenced_attributes

    # print('\n------')
    # print(X_train[0])
    # print('\n------')

    # print('\n------')
    # print(X_train_new[0][0])
    # print('\n------')

    # print(type(X_train[0]))
    # print(X_train[0].shape)
    # print(X_train[0])

    y_train = y_train.tolist()
    for i in range(len(y_train)):
        for _ in range(3):
            y_train[i] = np.append(y_train[i], y_train[i][0])
        y_train[i] = np.asarray(y_train[i])
        y_train[i] = y_train[i].reshape(4, 1)

    print(len(X_train))
    print(len(y_train))

    print(X_train[0])
    print(y_train[0])
    print(type(X_train[0]))
    print(type(y_train[0]))
    print('\n---------\n')

    # # generate input and embedding layers
    in_layers, em_layers = GRUModel.get_input_and_embedding_layers(X_train_enc, False)
    # build and compile model
    model = GRUModel.build(in_layers, em_layers)
    # # fit model
    GRUModel.fit(model, X_train_enc, X_test_enc, y_train_enc, y_test_enc)
