"""
    Name: model.py
    Purpose:

    @author Bartosz Świtalski, Piotr Frątczak

    Warsaw University of Technology
    Faculty of Electronics and Information Technology
"""
from model.model import build_hyper_model

import kerastuner as kt


def tune_params(X_train_enc, X_test_enc, y_train_enc, y_test_enc):
    # define hyper model
    tuner = kt.Hyperband(
        build_hyper_model, objective="val_accuracy", max_epochs=50, hyperband_iterations=5, overwrite=True,
        project_name='keras_tuner_config_output'
    )

    print(tuner.search_space_summary())
    tuner.search(X_train_enc, y_train_enc,
                 epochs=50,
                 validation_data=(X_test_enc, y_test_enc))

    best_model = tuner.get_best_models(1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
    tuner.results_summary()

    print(best_model)
    print(best_hyperparameters)
