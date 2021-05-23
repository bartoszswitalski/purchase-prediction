"""
    Name: aggregate.py
    Purpose: sessions.csv data aggregation for neural network input

    @author Bartosz Świtalski, Piotr Frątczak

    Warsaw University of Technology
    Faculty of Electronics and Information Technology
"""
import pandas as pd
import numpy as np


from preprocess.csv_read import get_csv_data


def get_aggregated_sessions():
    data = get_csv_data('output/', 'sessions_encoded.csv')
    dataset = data.values

    dataset = np.split(dataset, np.where(np.diff(dataset[:, 0]))[0] + 1)
    sessions = aggregate_sessions(dataset)


def aggregate_sessions(dataset):
    sessions = np.array([])

    for session in dataset:
        session = np.asarray(session)

        if session.shape[0] < 2:
            continue
        elif session.shape[0] < 4:
            session = np.split(session, [1])
        elif session.shape[0] < 7:
            session = np.split(session, [1, 3])
        elif session.shape[0] < 11:
            session = np.split(session, [1, 3, 6])
        elif session.shape[0] > 10:
            session = np.split(session, [1, 3, 6, 10])

        # np.append(sessions, session)
        # session = np.asarray(session)
        # print(session.shape)
        # print(session)
        # print('\n\n')

        # todo: later

    print(sessions)

    return sessions
