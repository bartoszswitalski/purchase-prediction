"""
    Name:       csv_ready.py
    Purpose:    CSV data preprosessing.

    @author Bartosz Świtalski, Piotr Frątczak

    Warsaw University of Technology
    Faculty of Electronics and Information Technology
"""
import pandas as pd


def get_csv_data(path, file_name):
    return pd.read_csv(path + file_name, sep=';')
