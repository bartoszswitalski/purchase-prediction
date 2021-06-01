"""
    Name: padding.py
    Purpose: sessions.csv data aggregation for neural network input

    @author Bartosz Świtalski, Piotr Frątczak

    Warsaw University of Technology
    Faculty of Electronics and Information Technology
"""
import pandas as pd
import numpy as np


from preprocess.csv_read import get_csv_data


def get_padded_sessions(directory, filename):
    data = get_csv_data(directory, filename)
    dataset = data.values

    dataset = np.split(dataset, np.where(np.diff(dataset[:, 0]))[0] + 1)

    padded_dataset = pad_sessions_with_zeros(dataset)
    # sessions = padded_dataset.transpose((2, 0, 1)).reshape(padded_dataset.shape[1], -1)
    sessions = padded_dataset.transpose((0, 2, 1)).reshape(-1, padded_dataset.shape[1])
    return sessions


def pad_sessions_with_zeros(dataset):
    dummy = np.array([0] * 10)

    for i in range(len(dataset)):
        if len(dataset[i]) < 11:
            for j in range(11 - len(dataset[i])):
                dataset[i] = np.append(dataset[i], dummy)
            dataset[i] = dataset[i].reshape(11, 10)
        dataset[i] = dataset[i].T

    dataset = np.asarray(dataset)
    return dataset
