"""
    Name:
    Purpose:

    @author Bartosz Świtalski, Piotr Frątczak

    Warsaw University of Technology
    Faculty of Electronics and Information Technology
"""
import datetime
import math

import json_lines
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.distutils.misc_util import is_string
from pandas.core.dtypes.common import is_numeric_dtype


def get_jsonl_data(file_name):
    data = []
    with open('data/' + file_name) as f:
        for item in json_lines.reader(f):
            data.append(item)

    return data


def products_check():
    df = pd.DataFrame(get_jsonl_data('products.jsonl'))
    cols = df.columns

    """
    Check if:
    1) id is integer (and is not null)
    2) id is unique
    3) id is in good range of values

    4) product name is not null
    5) product name is valid string
    
    6) category path is not null
    7) category path is valid string(semicolon separated)
    
    8) price is not null
    9) price is in valid range
    """
    print('\n------ products.jsonl check')

    print('{product_id}')
    check_unique_values(df, cols)
    check_if_numeric(df, 'product_id')
    check_range(df, 1001, 1319, 'product_id')

    print('\n{product_name}')
    check_if_string(df, 'product_name')
    check_if_empty(df, 'product_name')

    print('\n{category_path}')
    check_if_string(df, 'category_path')
    check_if_empty(df, 'category_path')

    print('\n{price}')
    check_if_float(df, 'price')
    check_range(df, 0, 10000, 'price')

    # check_plot(df)


def users_check():
    df = pd.DataFrame(get_jsonl_data('users.jsonl'))
    cols = df.columns

    """
    Check if:
    1) id is integer (and is not null)
    2) id is unique
    3) id is in good range of values
    """
    print('\n------ users.jsonl check')

    print('{user_id}')
    check_unique_values(df, cols)
    check_if_numeric(df, 'user_id')
    check_range(df, 102, 301, 'user_id')


def sessions_check():
    df_s = pd.DataFrame(get_jsonl_data('sessions.jsonl'))
    df_p = pd.DataFrame(get_jsonl_data('products.jsonl'))
    df_u = pd.DataFrame(get_jsonl_data('users.jsonl'))

    cols = df_s.columns

    users_ids = df_s['user_id']
    products_ids = df_s['product_id']
    """
    Check if:
    1) session_id is not null
    
    2) timestamp is of valid format (and not null)
    
    3) user_id is valid integer (and is not null)
    3a) {user_id} exists in users.jsonl
    
    4) product_id is not null
    4a) {product_id} exists in products.jsonl
    
    5) session_type is of valid format
    
    6) offered discount is valid integer (and is not null)
    
    7) {user_id} exists in users.jsonl
    """
    print('\n------ sessions.jsonl check')

    print(df_s)

    print('\n{session_id}')
    check_if_numeric(df_s, 'session_id')
    check_range(df_s, 100001, 110151, 'session_id')

    print('{timestamp}')
    check_timestamp(df_s, 'timestamp')

    print('{user_id}')
    check_nulls_overall(df_s)
    df_s = delete_nulls(df_s, 'product_id')

    df_s = fix_null_user_ids(df_s)
    df_s['user_id'] = df_s['user_id'].astype(int)

    check_constraints_users_ids(df_s, df_u)

    print('{product_id}')
    df_s['product_id'] = df_s['product_id'].astype(int)
    check_constraints_products_ids(df_s, df_p)

    print('{event_type}')
    check_event_type(df_s)

    print('{offered_discount}')
    df_s['offered_discount'] = df_s['offered_discount'].astype(int)
    check_if_numeric(df_s, 'offered_discount')


def check_unique_values(df, cols):
    print('------checking unique values')
    samples_num = df.shape[0]
    unique_num = df[cols[0]].unique().size
    print('#Samples: {} | #unique: {}'.format(samples_num, unique_num))
    print('---done checking if unique')


def check_range(df, left, right, col_name):
    print('------checking range')
    invalid_elem_sum = 0

    for index, row in df.iterrows():
        if isinstance(row[col_name], float) and (row[col_name] < left or row[col_name] > right):
            print('Value {} not in range | row: {}'.format(row[col_name], index + 1))
            invalid_elem_sum += 1

    print('# of invalid elements(range): {}'.format(invalid_elem_sum))
    print('---done checking range')


def check_if_numeric(df, col_name):
    print('------checking if numeric')
    for index, row in df.iterrows():
        if not isinstance(row[col_name], int):
            print(row[col_name])
            print('Wrong value at row: {}'.format(index + 1))
    print('---done checking if numeric')


def check_for_nulls(df, col_name):
    print('------checking for nulls')
    df_aux = df.isnull()
    for index, row in df_aux.iterrows():
        if row[col_name]:
            print('Null value at row: {}'.format(index + 1))
    print('---done checking for nulls')


def check_nulls_overall(df):
    print('------checking nulls overall')
    print(df.isnull().sum())
    print('---done checking nulls overall')


def delete_nulls(df, col_name):
    print('------deleting nulls')
    df = df[df[col_name].notna()]
    print(df.isnull().sum())
    print(df)
    print('---done deleting nulls')
    return df


def fix_null_user_ids(df):
    print('------fixing null users ids')
    df['user_id'] = df['user_id'].fillna(0)
    curr_session = 0
    curr_user_id = 0
    users_ids = {}
    for index, row in df.iterrows():
        if curr_session != row['session_id']:
            curr_session = row['session_id']
            curr_user_id = 0
        if row['user_id'] != 0 and row['user_id'] != curr_user_id:
            curr_user_id = row['user_id']
            users_ids[str(curr_session)] = curr_user_id
            # print([curr_session, curr_user_id])

    for index, row in df.iterrows():
        if row['user_id'] == 0:
            if str(row['session_id']) in users_ids.keys():
                # row['user_id'] = users_ids[str(row['session_id'])]
                df.at[index, 'user_id'] = users_ids[str(row['session_id'])]

    df = df[df.user_id != 0]
    print(df)
    print('---done fixing null users ids')

    return df


def check_constraints_users_ids(df_a, df_b):
    print('------checking constraints for user_id')
    df_aux = df_a
    df_aux = df_aux.assign(integrity=df_a.user_id.isin(df_b.user_id).astype(int))
    result = df_aux[df_aux['integrity'] == 0].index.tolist()
    print(result)
    print('---done checking constraints for user_id')


def check_constraints_products_ids(df_a, df_b):
    print('------checking constraints for product_id')
    df_aux = df_a
    df_aux = df_aux.assign(integrity=df_a.product_id.isin(df_b.product_id).astype(int))
    result = df_aux[df_aux['integrity'] == 0].index.tolist()
    print(result)
    print('---done checking constraints for product_id')


def check_event_type(df):
    print('------checking event types frequencies')
    count = 0
    for index, row in df.iterrows():
        if row['event_type'] != 'VIEW_PRODUCT' and row['event_type'] != 'BUY_PRODUCT':
            count += 1
    print('Wrong event types number: {}'.format(count))

    df_aux = df.groupby('event_type').count()
    print(df_aux)
    print('---done checking event types frequencies')


def check_if_float(df, col_name):
    print('---checking if float')
    for index, row in df.iterrows():
        if not isinstance(row[col_name], float):
            print('Wrong value at row: {}'.format(index + 1))
    print('---done checking if float')


def check_if_string(df, col_name):
    print('---checking if string')
    for index, row in df.iterrows():
        if not isinstance(row[col_name], str):
            print('Wrong value at row: {}'.format(index + 1))
    print('---done checking if string')


def check_if_empty(df, col_name):
    print('---checking if empty')
    for index, row in df.iterrows():
        if not row[col_name]:
            print('Empty value at row: {}'.format(index + 1))
    print('---done checking if empty')


def check_timestamp(df, col_name):
    print('---checking date format')
    for index, row in df.iterrows():
        if not valid_timestamp(row[col_name]):
            print('Invalid date at row: {}'.format(index + 1))
    print('---done checking date format')


def valid_timestamp(timestamp_str):
    date = timestamp_str[:10]
    sep = timestamp_str[10]
    time = timestamp_str[11:]

    if sep != 'T':
        return False

    try:
        datetime.datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        return False

    try:
        datetime.datetime.strptime(time, '%H:%M:%S')
        return True
    except ValueError:
        return False


def check_plot(df):
    print('---plot')
    cols = df.columns
    plt.boxplot(df[cols[3]])
    plt.yscale('log')
    plt.show()
