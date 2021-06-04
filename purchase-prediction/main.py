"""
    Name: main.py
    Purpose:

    @author Bartosz Świtalski, Piotr Frątczak

    Warsaw University of Technology
    Faculty of Electronics and Information Technology
"""
# from preprocess.check_users import users_check
# from preprocess.check_products import products_check
# from preprocess.check_sessions import sessions_check

from model.simple.deep_nn import DeepModel
from model.gru.deep_nn_gru import GRUModel
from model.gpu_setup.usage import limit_gpu_usage

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
    # limit GPU usage
    limit_gpu_usage()

    """
        uncomment to fit basic model
    """
    # get train and test sets
    X_train_enc, X_test_enc, y_train_enc, y_test_enc = DeepModel.get_dataset()
    # generate input and embedding layers
    in_layers, em_layers = DeepModel.get_input_and_embedding_layers(X_train_enc)
    # build model
    model = DeepModel.build(in_layers, em_layers)
    # fit model
    DeepModel.fit(model, X_train_enc, X_test_enc, y_train_enc, y_test_enc)
    # test prediction
    DeepModel.test_predict(model)

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
    # X_train_enc, _, _, _ = DeepModel.get_dataset()
    # # get train and test sets
    # X_train, X_test, y_train, y_test = GRUModel.get_dataset()
    # # generate input and embedding layers
    # in_layers, em_layers = GRUModel.get_input_and_embedding_layers(X_train_enc, True)
    # # build and compile model
    # model = GRUModel.build(in_layers, em_layers)
    # # fit model
    # GRUModel.fit(model, X_train, X_test, y_train, y_test)
