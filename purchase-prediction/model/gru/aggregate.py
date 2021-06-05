import pandas as pd
import numpy as np

from utils.csv_read import get_csv_data


def get_aggregated_sessions():
    data = get_csv_data('output/', 'sessions_test.csv')
    data = data.drop_duplicates()
    data = data.values

    data = np.split(data, np.where(np.diff(data[:, 0]))[0] + 1)
    sessions = aggregate_sessions(data)


def aggregate_sessions(dataset):
    # sessions = np.array([])
    sessions = list()

    for session in dataset:
        i = 0
        if session.shape[0] < 2:
            pass
        elif session.shape[0] < 4:
            session = np.split(session, [1])
        elif session.shape[0] < 7:
            session = np.split(session, [1, 3])
        elif session.shape[0] < 11:
            session = np.split(session, [1, 3, 6])

        for entry in session:
            entry = aggregate_entry(entry)
        #
        # print(session)
        # print(type(session))
        # np.append(sessions, session)
        sessions.append(session)
        # session = np.asarray(session)
        # print(session.shape)
        # print(session)
        # print('\n\n')

        # todo: later

    # print(sessions[0])

    return sessions


def aggregate_entry(entry):
    return entry
