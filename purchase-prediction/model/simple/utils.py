import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils.csv_read import get_csv_data


def load_dataset(directory, filename):
    # load the dataset as a pandas DataFrame
    data = get_csv_data(directory, filename)
    # drop session_id column
    data = data.drop(['session_id'], axis=1)
    # retrieve numpy array
    dataset = data.values
    print(dataset)
    print(dataset.shape)
    # split into input (X) and output (Y) variables
    X = dataset[:, :-1]
    Y = dataset[:, -1]
    # reshape target to be a 2d array
    Y = Y.reshape((len(Y), 1))

    return X, Y


def prepare_inputs(X_train, X_test):
    X_train_enc, X_test_enc = list(), list()
    # append first column that will not be encoded
    X_train_enc.append(X_train[:, 0])
    X_test_enc.append(X_test[:, 0])
    # label encode every other column
    for i in range(1, X_train.shape[1]):
        le = LabelEncoder()
        le.fit(X_train[:, i])
        # encode
        train_enc = le.transform(X_train[:, i])
        test_enc = le.transform(X_test[:, i])
        # store
        X_train_enc.append(train_enc)
        X_test_enc.append(test_enc)

    X_train_enc[0] = X_train_enc[0].astype('float32')
    X_test_enc[0] = X_test_enc[0].astype('float32')

    return X_train_enc, X_test_enc


def prepare_targets(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    return y_train_enc, y_test_enc
