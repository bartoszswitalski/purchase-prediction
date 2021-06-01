"""
    Name: main.py
    Purpose:

    @author Bartosz Świtalski, Piotr Frątczak

    Warsaw University of Technology
    Faculty of Electronics and Information Technology
"""
from model.model import get_sequential_dataset
# from preprocess.check_users import users_check
# from preprocess.check_products import products_check
# from preprocess.check_sessions import sessions_check
# from preprocess.prepare_dataset import load_dataset, prepare_inputs, prepare_targets
from model.model import build_model, build_sequence_model, get_dataset, get_input_and_embedding_layers, fit_model, get_input_and_embedding_layers_for_sequential
from model.hyper_tuning import tune_params
# from model.gpu_setup import limit_gpu_usage

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
    # X_train_enc, X_test_enc, y_train_enc, y_test_enc = get_dataset()
    # # generate input and embedding layers
    # in_layers, em_layers = get_input_and_embedding_layers(X_train_enc)
    # # fit model
    # fit_model(X_train_enc, X_test_enc, y_train_enc, y_test_enc, em_layers, in_layers)

    """
        uncomment to tune parameters 
    """
    # # get train and test sets
    # X_train_enc, X_test_enc, y_train_enc, y_test_enc = get_dataset()
    # # tune parameters
    # tune_params(X_train_enc, X_test_enc, y_train_enc, y_test_enc)

    """
        uncomment to build sequence model
    """
    # get X_train_enc for vocabulary
    X_train_enc, _, _, _ = get_dataset()
    # generate input and embedding layers for sequence model with mask_zero set to True
    in_layers, em_layers = get_input_and_embedding_layers_for_sequential(X_train_enc, True)
    # get train and test sets
    # tutaj patrz po reshape
    X_train, X_test, y_train, y_test = get_sequential_dataset()
    # # define sequence model
    sequence_model = build_sequence_model(in_layers, em_layers)

    print(X_train)
    print(y_train)
    sequence_model.fit(X_train, y_train, epochs=10, batch_size=128, verbose=2)
