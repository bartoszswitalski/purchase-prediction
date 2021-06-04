import numpy as np
import tensorflow as tf
import kerastuner as kt

from tensorflow import keras
from sklearn.model_selection import train_test_split

from model.lstm.utils import load_sequential_dataset, reshape_sequential_inputs


class LSTMModel:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset():
        # load the dataset
        X, y = load_sequential_dataset('output/', 'sessions_encoded.csv')

        # split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, shuffle=False)

        # prepare data
        X_train, X_test = reshape_sequential_inputs(X_train, X_test)

        # make output 3d
        y_train = y_train.reshape((len(y_train), 1, 1))
        y_test = y_test.reshape((len(y_test), 1, 1))

        print(X_train)
        print(X_test)
        print(y_train)
        print(y_test)

        return X_train, X_test, y_train, y_test
