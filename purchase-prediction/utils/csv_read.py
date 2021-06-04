import pandas as pd


def get_csv_data(path, file_name):
    return pd.read_csv(path + file_name, sep=';')
