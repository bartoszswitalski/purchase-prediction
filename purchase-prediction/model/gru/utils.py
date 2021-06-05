import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils.csv_read import get_csv_data
from model.gru.aggregate import get_aggregated_sessions


def load_sequential_dataset(directory, filename):
    # # load the dataset as a pandas DataFrame
    # data = get_csv_data(directory, filename)
    # # drop duplicated session entries
    # data = data.drop_duplicates()
    # load the dataset as pandas DataFrame having aggregated it beforehand
    data = get_aggregated_sessions(directory, filename)
    # save to csv
    data.to_csv('output/data_aggregated_without_duplicates', sep=';', encoding='utf-8', index=False)
    # get padded sessions data
    data_padded = pad_data(data)
    # delete duplicates
    # retrieve numpy array
    dataset = data_padded.values

    # split into input (X) and output (Y) variables
    X = dataset[:, :-1]
    Y = dataset[:, -1]
    # reshape target to be a 2d array
    X = X.reshape(-1, 4, 8)
    print('\n---------\n')
    print(X.shape)
    print('\n---------\n')

    Y = Y.reshape((len(Y), 1))

    return X, Y


def pad_data(data):
    dataset = data.values

    dataset = np.split(dataset, np.where(np.diff(dataset[:, 0]))[0] + 1)

    # dummy = np.array([0] * 8)
    dummy = [0] * 8

    for i in range(len(dataset)):
        y = dataset[i][0][-1]
        dataset[i] = dataset[i].tolist()
        for j in range(len(dataset[i])):
            dataset[i][j] = dataset[i][j][1:-1]
        if len(dataset[i]) < 4:
            for _ in range(4 - len(dataset[i])):
                dataset[i].append(dummy)
        dataset[i] = np.append(dataset[i], y)

    dataset = np.asarray(dataset)

    df = pd.DataFrame(dataset)
    df.to_csv('output/sessions_sequenced.csv', sep=';', encoding='utf-8', index=False)

    return df


def prepare_inputs_for_sequential(X_train, X_test):
    X_train_enc, X_test_enc = list(), list()

    # append first column that will not be encoded
    for i in range(4):
        X_train_enc.append(X_train[:, i * 8])
        X_train_enc.append(X_train[:, 1 + i * 8])

        X_test_enc.append(X_test[:, i * 8])
        X_test_enc.append(X_test[:, 1 + i * 8])

        # label encode every other column
        for j in range(2, 8):
            le = LabelEncoder()
            le.fit(X_train[:, i * 8 + j])
            # encode
            train_enc = le.transform(X_train[:, i * 8 + j])
            test_enc = le.transform(X_test[:, i * 8 + j])
            # store
            X_train_enc.append(train_enc)
            X_test_enc.append(test_enc)

        X_train_enc[i * 8] = X_train_enc[i * 8].astype('float32')
        X_train_enc[1 + i * 8] = X_train_enc[1 + i * 8].astype('float32')
        X_test_enc[i * 8] = X_test_enc[i * 8].astype('float32')
        X_test_enc[1 + i * 8] = X_test_enc[1 + i * 8].astype('float32')

    return X_train_enc, X_test_enc
