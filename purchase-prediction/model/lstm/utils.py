import numpy as np
import pandas as pd

from utils.csv_read import get_csv_data


def load_sequential_dataset(directory, filename):
    # load the dataset as a pandas DataFrame
    data = get_csv_data(directory, filename)
    # get padded sessions data
    data_padded = pad_data(data)
    # drop session_id column
    data_padded = data_padded.drop(['session_id'], axis=1)
    # retrieve numpy array
    dataset = data_padded.values
    print(dataset)
    print(dataset.shape)
    # split into input (X) and output (Y) variables
    X = dataset[:, :-1]
    Y = dataset[:, -1]
    # reshape target to be a 2d array
    Y = Y.reshape((len(Y), 1))

    return X, Y


def pad_data(data):
    dataset = data.values

    dataset = np.split(dataset, np.where(np.diff(dataset[:, 0]))[0] + 1)

    dummy = np.array([0] * 10)

    for i in range(len(dataset)):
        if len(dataset[i]) < 11:
            for j in range(11 - len(dataset[i])):
                dataset[i] = np.append(dataset[i], dummy)
            dataset[i] = dataset[i].reshape(11, 10)
        # dataset[i] = dataset[i].T

    dataset = np.asarray(dataset)

    # sessions = padded_dataset.transpose((2, 0, 1)).reshape(padded_dataset.shape[1], -1)
    # sessions = padded_dataset.transpose((0, 2, 1)).reshape(-1, padded_dataset.shape[1])
    dataset = dataset.reshape(-1, dataset.shape[2])

    df = pd.DataFrame(dataset, columns=['session_id', 'price', 'offered_discount', 'category_path',
                                        'city', 'month', 'day', 'weekDay', 'hour', 'is_buy'])
    # df.to_csv('output/sessions_sequence_len_11.csv', sep=';', encoding='utf-8', index=False)
    # print(df)

    return df


def reshape_sequential_inputs(X_train, X_test):
    return X_train.T, X_test.T


def reshape_sequential_targets(y_train, y_test):
    return y_train, y_test
