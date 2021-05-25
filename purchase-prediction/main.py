"""
    Name: main.py
    Purpose:

    @author Bartosz Świtalski, Piotr Frątczak

    Warsaw University of Technology
    Faculty of Electronics and Information Technology
"""
# from preprocess.prepare_dataset import load_dataset, prepare_inputs, prepare_targets
from model.model import build_model, get_dataset, get_input_and_embedding_layers, fit_model
from model.hyper_tuning import tune_params
from model.gpu_setup import limit_gpu_usage

if __name__ == '__main__':

    # limit GPU usage
    limit_gpu_usage()
    # get train and test sets
    X_train_enc, X_test_enc, y_train_enc, y_test_enc = get_dataset()
    # generate input and embedding layers
    in_layers, em_layers = get_input_and_embedding_layers(X_train_enc)
    # define model
    model = build_model(em_layers, in_layers)
    # fit model
    # fit_model(X_train_enc, X_test_enc, y_train_enc, y_test_enc, em_layers, in_layers)

    # # tune parameters
    tune_params(X_train_enc, X_test_enc, y_train_enc, y_test_enc)
