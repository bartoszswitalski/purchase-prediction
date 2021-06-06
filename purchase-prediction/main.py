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

    """
        uncomment to build sequence model
    """
    X_train_enc, X_test_enc, y_train_enc, y_test_enc = DeepModel.get_dataset()
    # get train and test sets
    X_train, X_test, y_train, y_test = GRUModel.get_dataset()

    x1 = X_train[:, :, 0]
    x2 = X_train[:, :, 1]
    x3 = X_train[:, :, 2]
    x4 = X_train[:, :, 3]
    x5 = X_train[:, :, 4]
    x6 = X_train[:, :, 5]
    x7 = X_train[:, :, 6]
    x8 = X_train[:, :, 7]

    xt1 = X_test[:, :, 0]
    xt2 = X_test[:, :, 1]
    xt3 = X_test[:, :, 2]
    xt4 = X_test[:, :, 3]
    xt5 = X_test[:, :, 4]
    xt6 = X_test[:, :, 5]
    xt7 = X_test[:, :, 6]
    xt8 = X_test[:, :, 7]

    # # generate input and embedding layers
    in_layers, em_layers = GRUModel.get_input_and_embedding_layers(X_train_enc)
    # build and compile model
    model = GRUModel.build(in_layers, em_layers)
    # # fit model
    # GRUModel.fit(model, [x1, x2, x3, x4, x5, x6, x7, x8], [xt1, xt2, xt3, xt4, xt5, xt6, xt7, xt8], y_train, y_test)
    GRUModel.fit(model, X_train_enc, X_test_enc, y_train_enc, y_test_enc)
