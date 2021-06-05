import pandas as pd
import numpy as np

from utils.csv_read import get_csv_data


def get_aggregated_sessions(directory, filename):
    data = get_csv_data(directory, filename)
    data = data.drop_duplicates()
    data = data.values

    data = np.split(data, np.where(np.diff(data[:, 0]))[0] + 1)

    return aggregate_sessions(data)


def aggregate_sessions(dataset):
    # sessions = np.array([])
    sessions = list()

    for session in dataset:
        i = 0
        if session.shape[0] < 2:
            session = np.split(session, [0])
            session = session[1:]
        elif session.shape[0] < 4:
            session = np.split(session, [1])
        elif session.shape[0] < 7:
            session = np.split(session, [1, 3])
        elif session.shape[0] < 11:
            session = np.split(session, [1, 3, 6])

        for i in range(len(session)):
            session[i] = aggregate_to_single_entry(session[i])
        sessions.append(session)

    dataset = list()
    for session_data in sessions:
        for entry in session_data:
            dataset.append(entry)

    return pd.DataFrame(dataset)


def aggregate_to_single_entry(entry):
    new_entry = [0] * 10
    length = len(entry)

    # print(entry)

    # session_id
    new_entry[0] = entry[0][0]
    # city
    new_entry[4] = entry[0][4]
    # month
    new_entry[5] = entry[0][5]
    # day
    new_entry[6] = entry[0][6]
    # weekDay
    new_entry[7] = entry[0][7]
    # target (y)
    new_entry[9] = entry[0][9]

    # price, discount and hour is a mean
    for i in range(length):
        new_entry[1] += entry[i][1]
        new_entry[2] += entry[i][2]
        new_entry[8] += entry[i][8]
    # round to int
    new_entry[1] = round(new_entry[1] / length)
    new_entry[2] = round(new_entry[2]/length)
    new_entry[8] = round(new_entry[8]/length)

    # category is the most frequent one
    categories = []
    for i in range(length):
        categories.append(entry[i][3])

    new_category = max(set(categories), key=categories.count)
    # print(new_category)
    new_entry[3] = new_category

    return new_entry
