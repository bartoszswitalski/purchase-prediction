"""
    Name: utils.py
    Purpose: utility functions for various data checks

    @author Bartosz Świtalski, Piotr Frątczak

    Warsaw University of Technology
    Faculty of Electronics and Information Technology
"""
from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype

import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
import datetime

from sklearn import feature_selection

SEED = 87342597


def check_if_empty(df, column_name):
    print('------checking if empty')
    print(df[column_name].notnull().all() and (df[column_name] != '').all())

    if df[column_name].isnull().any() and (df[column_name] != '').any():
        print('Sum of empty elements: {}'.format(df[(df[column_name].isnull()) | (df[column_name] == '')].count()[0]))


def check_if_unique_values(df, column_name):
    print('------checking unique values')
    print(df[column_name].is_unique)


def check_if_numeric(df, column_name):
    print('------checking if numeric')
    print(is_numeric_dtype(df[column_name]))


def check_range(df, left, right, column_name):
    print('------checking range: [{};{}]'.format(left, right))
    print(df[column_name].between(left, right, inclusive=True).all())

    if not df[column_name].between(left, right, inclusive=True).all():
        print('sum of invalid elements: {}'.format(
            df[(df[column_name] > right) | (df[column_name] < left)].count()[0]))


def plot_histogram(df, column_name):
    print('------' + column_name + ' histogram')
    df[column_name].value_counts().plot(kind='bar')
    plt.show()


def check_if_string(df, column_name):
    print('------checking if string')
    print(is_string_dtype(df[column_name]))


def check_if_float(df, column_name):
    print('------checking if float')
    print(df[column_name].dtype == 'float64')


def check_boxplot(df, column_name):
    print('------plot')
    plt.boxplot(df[column_name])
    plt.yscale('log')

    plt.savefig('output/products_jsonl_' + column_name + '_boxplot.jpg')
    print('saved to output')


def delete_invalid_values(df, column_name, lower_bound, upper_bound):
    print('------deleting invalid prices')
    return df[(df[column_name] > lower_bound) & (df[column_name] < upper_bound)]


def check_timestamp(df, column_name):
    print('------checking date format')
    print(df[column_name].apply(valid_timestamp).all())


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


def check_event_type(df):
    print('------checking if event types valid')
    print(df['event_type'].apply(valid_event_type).all())

    print('\n# of elements')
    df = df.groupby('event_type')['event_type'].count()
    print(df)


def valid_event_type(string):
    if string == 'VIEW_PRODUCT' or string == 'BUY_PRODUCT':
        return True
    else:
        return False


def delete_nulls(df, column_name):
    print('------deleting ' + column_name + ' nulls')
    df = df[df[column_name].notna()]
    print('done')
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
    print('done')
    return df


def check_constraints(df_a, df_b, column_name):
    print('------checking constraints for ' + column_name)
    df_aux = df_a.copy()
    df_aux = df_aux.assign(integrity=df_a[column_name].isin(df_b[column_name]).astype(int))
    result = df_aux[df_aux['integrity'] == 0].index.tolist()
    if not result:
        print(True)
    else:
        print(str(len(result)) + " constraint violations.")


def delete_constraints_violations(df_s, df_p, column_name):
    print('------deleting constraints violations')
    df_s = df_s.join(df_p.set_index(column_name), on=column_name)
    df_s = df_s.dropna()

    return df_s


def count_sessions_by_purchase(df):
    print('------counting sessions by purchase')
    buy_sessions = 0
    view_sessions = -1

    current_session_id = -1
    finished_with_purchase = 0
    for index, row in df.iterrows():
        if current_session_id != row['session_id']:
            buy_sessions += finished_with_purchase
            view_sessions += 1 - finished_with_purchase

            current_session_id = row['session_id']
            finished_with_purchase = 0

        if row['event_type'] == 'BUY_PRODUCT':
            finished_with_purchase = 1

    print('Number of sessions with a purchase: {:>10}'.format(buy_sessions))
    print('Number of sessions without a purchase: {:>7}'.format(view_sessions))
    print('{}% of sessions end with a purchase.'.format(round(buy_sessions / (buy_sessions + view_sessions), 4) * 100))


def add_is_buy(df_s):

    buy_sessions = []
    current_session_id = -1
    for index, row in df_s.iterrows():
        if current_session_id != row['session_id']:
            current_session_id = row['session_id']

        if row['event_type'] == 'BUY_PRODUCT':
            buy_sessions.append(current_session_id)

    df_s = df_s.drop(['event_type'], axis=1)

    return df_s.assign(is_buy=np.where(df_s['session_id'].isin(buy_sessions), '1', '0'))


def session_mutual_info_for_input(df):
    print('{sessions is_buy mutual information scores}')

    cols = df.columns.tolist()
    cols.remove('is_buy')
    cols.remove('session_id')
    df_x = df[cols].copy()
    df_y = df[['is_buy']].copy()

    df_x = df_x[['price', 'offered_discount', 'category_path', 'city', 'month', 'day', 'weekDay', 'hour']]

    # df_x = pd.get_dummies(df_x, columns=['weekDay'])
    # cols = df_x.columns.tolist()

    # with pd.option_context('display.max_columns', None):
    #     print(df_x)

    mis = feature_selection.mutual_info_classif(df_x, df_y.values.flatten().reshape(-1, ),
                                                discrete_features=[1, 2, 3, 4, 5, 6, 7]).tolist()

    noises_sum = [0, 0, 0, 0, 0, 0, 0, 0]
    average_over = 100

    for i in range(average_over):
        df_y['is_buy'] = np.random.permutation(df_y['is_buy'].values)
        noises = feature_selection.mutual_info_classif(df_x, df_y.values.flatten().reshape(-1, ),
                                                       discrete_features=[1, 2, 3, 4, 5, 6, 7]).tolist()

        noises_sum = [a + b for a, b in zip(noises, noises_sum)]  # add lists element-wise
    noises_avg = [a / average_over for a in noises_sum]

    df_heatmap = pd.DataFrame({'is_buy': mis, 'noise': noises_avg}, index=cols)

    plt.subplots(figsize=(8, 6))
    sb.heatmap(df_heatmap, annot=True)
    plt.savefig("output/sessions_to_output_mutual_info.jpg")
    print('saved to output')