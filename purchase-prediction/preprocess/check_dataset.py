"""
    Name:
    Purpose:

    @author Bartosz Świtalski, Piotr Frątczak

    Warsaw University of Technology
    Faculty of Electronics and Information Technology
"""
from preprocess.json_read import *

import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype
from sklearn import feature_selection, metrics, preprocessing

SEED = 87342597


def users_check():
    df = pd.DataFrame(get_jsonl_data('users.jsonl'))
    """
    Check if:
    1) id is an integer (and is not null)
    2) id is unique
    3) id is in good range of values
    """
    print('###### users.jsonl check ######')

    print('{user_id}')
    check_if_empty(df, 'user_id')
    check_unique_values(df, 'user_id')
    check_if_numeric(df, 'user_id')
    check_range(df, 102, 301, 'user_id')

    df = df.drop(['name', 'city', 'street'], axis=1)
    df.to_csv("output/users.csv", sep=';', encoding='utf-8', index=False)

    return df


def products_check():
    df = pd.DataFrame(get_jsonl_data('products.jsonl'))
    """
    Check if:
    1) id is an integer (and is not null)
    2) id is unique
    3) id is in good range of values

    4) product name is not null
    5) product name is a valid string

    6) category path is not null
    7) category path is a valid string

    8) price is not null
    9) price is in valid range
    """
    print('\n###### products.jsonl check ######')

    print('{product_id}')
    check_if_empty(df, 'product_id')
    check_unique_values(df, 'product_id')
    check_if_numeric(df, 'product_id')
    check_range(df, 1001, 1319, 'product_id')

    print('\n{product_name}')
    check_if_empty(df, 'product_name')
    check_if_string(df, 'product_name')

    print('\n{category_path}')
    check_if_empty(df, 'category_path')
    check_if_string(df, 'category_path')

    print('\n{price}')
    check_if_empty(df, 'price')
    check_if_float(df, 'price')
    check_range(df, 0, 10 ** 4, 'price')
    check_plot(df, 'price')

    # products_mutual_info(df)

    df = delete_invalid_price(df)
    df = df.drop(['product_name'], axis=1)
    df.to_csv("output/products.csv", sep=';', encoding='utf-8', index=False)

    return df


def sessions_check():
    df_s = pd.DataFrame(get_jsonl_data('sessions.jsonl'))
    df_p = pd.DataFrame(get_jsonl_data('products.jsonl'))
    df_u = pd.DataFrame(get_jsonl_data('users.jsonl'))

    df_p = delete_invalid_price(df_p)
    df_p = df_p.drop(['product_name'], axis=1)

    """
    Check if:
    1) session_id is not null

    2) timestamp is of valid format (and not null)

    3) user_id is a valid integer (and is not null)
    3a) {user_id} exists in users.jsonl

    4) product_id is not null
    4a) {product_id} exists in products.jsonl

    5) session_type is of valid format

    6) offered discount is a valid integer (and is not null)

    7) {user_id} exists in users.jsonl
    """
    print('\n###### sessions.jsonl check ######')

    print('{session_id}')
    check_if_empty(df_s, 'session_id')
    check_if_numeric(df_s, 'session_id')
    check_range(df_s, 100001, 110151, 'session_id')

    print('{timestamp}')
    check_if_empty(df_s, 'timestamp')
    check_timestamp(df_s, 'timestamp')

    print('{user_id}')
    check_if_empty(df_s, 'user_id')

    print('{product_id}')
    check_if_empty(df_s, 'product_id')

    print('{event_type}')
    check_if_empty(df_s, 'event_type')
    check_event_type(df_s)
    count_sessions_by_purchase(df_s)

    print('{offered_discount}')
    check_if_empty(df_s, 'offered_discount')
    check_if_numeric(df_s, 'offered_discount')
    check_range(df_s, 0, 100, 'offered_discount')

    session_mutual_info(df_s)

    print('{after checking if missing data is MCAR}')
    print('\nClean missing data')

    df_s = delete_nulls(df_s, 'product_id')
    df_s = fix_null_user_ids(df_s)

    df_s['user_id'] = df_s['user_id'].astype(int)
    check_constraints_users_ids(df_s, df_u)

    df_s['product_id'] = df_s['product_id'].astype(int)
    check_constraints_products_ids(df_s, df_p)

    df_s = delete_constraints_violations(df_s, df_p)
    check_event_type(df_s)
    count_sessions_by_purchase(df_s)

    df_s['offered_discount'] = df_s['offered_discount'].astype(int)

    df_s.to_csv("output/sessions.csv", sep=';', encoding='utf-8', index=False)

    le = preprocessing.LabelEncoder()
    df_s['category_path'] = le.fit_transform(df_s['category_path'].values)
    df_s['month'] = pd.DatetimeIndex(df_s['timestamp']).month
    df_s['day'] = pd.DatetimeIndex(df_s['timestamp']).day
    df_s['weekDay'] = pd.DatetimeIndex(df_s['timestamp']).weekday
    df_s['hour'] = pd.DatetimeIndex(df_s['timestamp']).hour

    df_s = df_s.drop(['timestamp'], axis=1)

    df_s = add_is_buy(df_s)

    with pd.option_context('display.max_columns', None):
        print(df_s)

    session_mutual_info_for_input(df_s)


def check_unique_values(df, column_name):
    print('------checking unique values')
    print(df[column_name].is_unique)


def check_range(df, left, right, column_name):
    print('------checking range: [{};{}]'.format(left, right))
    print(df[column_name].between(left, right, inclusive=True).all())

    if not df[column_name].between(left, right, inclusive=True).all():
        print('sum of invalid elements: {}'.format(df[(df[column_name] > right) | (df[column_name] < left)].count()[0]))


def check_if_numeric(df, column_name):
    print('------checking if numeric')
    print(is_numeric_dtype(df[column_name]))


def delete_nulls(df, column_name):
    print('------deleting ' + column_name + ' nulls')
    df = df[df[column_name].notna()]
    # print(df.isnull().sum())
    print('done')
    return df


def delete_invalid_price(df):
    print('------deleting invalid prices')
    return df[(df['price'] > 0) & (df['price'] < 10 ** 4)]


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


def check_constraints_users_ids(df_a, df_b):
    print('------checking constraints for user_id')
    df_aux = df_a.copy()
    df_aux = df_aux.assign(integrity=df_a.user_id.isin(df_b.user_id).astype(int))
    result = df_aux[df_aux['integrity'] == 0].index.tolist()
    if not result:
        print(True)
    else:
        print(str(len(result)) + " constraint violations.")


def check_constraints_products_ids(df_a, df_b):
    print('------checking constraints for product_id')
    df_aux = df_a.copy()
    df_aux = df_aux.assign(integrity=df_a.product_id.isin(df_b.product_id).astype(int))
    result = df_aux[df_aux['integrity'] == 0].index.tolist()

    if not result:
        print(True)
    else:
        print(str(len(result)) + " constraint violations.")


def delete_constraints_violations(df_s, df_p):
    print('------deleting constraints violations')

    df_s = df_s.join(df_p.set_index('product_id'), on='product_id')
    df_s = df_s.drop(['purchase_id'], axis=1)
    df_s = df_s.dropna()

    return df_s


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


def check_if_float(df, column_name):
    print('------checking if float')
    print(df[column_name].dtype == 'float64')


def check_if_string(df, column_name):
    print('------checking if string')
    print(is_string_dtype(df[column_name]))


def check_if_empty(df, column_name):
    print('------checking if empty')
    print(df[column_name].notnull().all() and (df[column_name] != '').all())

    if df[column_name].isnull().any() and (df[column_name] != '').any():
        print('Sum of empty elements: {}'.format(df[(df[column_name].isnull()) | (df[column_name] == '')].count()[0]))


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


def check_plot(df, column_name):
    print('------plot')
    cols = df.columns
    plt.boxplot(df[column_name])
    plt.yscale('log')

    plt.savefig('output/products_jsonl_' + column_name + '_boxplot.jpg')
    print('saved to output')


def add_is_buy(df_s):

    buy_sessions = []
    current_session_id = -1
    for index, row in df_s.iterrows():
        if current_session_id != row['session_id']:
            current_session_id = row['session_id']

        if row['event_type'] == 'BUY_PRODUCT':
            buy_sessions.append(current_session_id)

    return df_s.assign(is_buy=np.where(df_s['session_id'].isin(buy_sessions), '1', '0'))


def session_mutual_info_for_input(df):
    print('{sessions is_buy mutual information scores}')

    cols = df.columns.tolist()
    cols.remove('is_buy')
    df_X = df[cols].copy()
    df_X = df_X.assign(event=np.where(df_X['event_type'] == 'BUY_PRODUCT', '1', '0'))
    df_X.drop(['event_type'], axis=1, inplace=True)
    df_X.rename(columns={'event': 'event_type'}, inplace=True)
    df_y = df[['is_buy']].copy()

    mis = feature_selection.mutual_info_classif(df_X, df_y.values.flatten().reshape(-1, ),
                                                discrete_features=[1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1]).tolist()

    df_heatmap = pd.DataFrame({'is_buy': mis}, index=cols)

    plt.subplots(figsize=(8, 6))
    sns.heatmap(df_heatmap, annot=True)
    plt.savefig("output/sessions_to_output_mutual_info.jpg")
    print('saved to output')


def session_mutual_info(df):
    print('{sessions user_id mutual information scores}')
    display_mutual_info(df, 'user_id', 'product_id')

    print('{sessions product_id mutual information scores}')
    display_mutual_info(df, 'product_id', 'user_id')


def mutual_info_score(df, col1, col2):
    crosstab = pd.crosstab(df[col1], df[col2], margins=False)
    return metrics.mutual_info_score(None, None, contingency=crosstab)


def mutual_info_score_with_noise(df, col1, col2, evaluations=100):
    mi = mutual_info_score(df, col1, col2)

    df_shuffled = df.copy()
    to_drop = df_shuffled.columns.tolist()
    to_drop.remove(col1)
    to_drop.remove(col2)
    df_shuffled.drop(to_drop, axis=1, inplace=True)

    np.random.seed(SEED)
    noise_mi = 0
    for i in range(evaluations):
        df_shuffled[col2] = np.random.permutation(df[col2].values)
        noise_mi += mutual_info_score(df_shuffled, col1, col2)

    return mi, noise_mi / evaluations


def display_mutual_info(df, column_name, column_w_nan):
    original_rest_cols = ['event_type', 'offered_discount']
    df_X = df[original_rest_cols].copy()
    df_X = df_X.assign(is_buy=np.where(df_X['event_type'] == 'BUY_PRODUCT', '1', '0'))
    df_X.drop(['event_type'], axis=1, inplace=True)
    rest_columns = ['is_buy', 'offered_discount']

    df_y = df[[column_name]].copy()
    df_y = df_y.assign(checkMCAR=np.where(df_y[column_name].isnull(), '1', '0'))
    df_y = df_y[['checkMCAR']].copy()

    df_na = df[[column_name, column_w_nan, rest_columns[1]]].copy()
    df_na = df_na.dropna(subset=[column_w_nan])
    df_y2 = df_na.assign(checkMCAR=np.where(df_na[column_name].isnull(), '1', '0'))
    df_y2 = df_y2[['checkMCAR']].copy()
    df_X2 = df_na[[column_w_nan, rest_columns[1]]].copy()

    mis = feature_selection.mutual_info_classif(df_X, df_y.values.flatten().reshape(-1, ),
                                                discrete_features=[1, 0]).tolist()
    mis2 = feature_selection.mutual_info_classif(df_X2, df_y2.values.flatten().reshape(-1, ),
                                                 discrete_features=[1, 0]).tolist()
    mis.append(mis2[0])

    noises_sum = [0, 0, 0]

    np.random.seed(SEED)
    average_over = 100
    for i in range(average_over):
        df_y['checkMCAR'] = np.random.permutation(df_y['checkMCAR'].values)
        df_y2['checkMCAR'] = np.random.permutation(df_y2['checkMCAR'].values)
        noises = feature_selection.mutual_info_classif(df_X, df_y.values.flatten().reshape(-1, ),
                                                       discrete_features=[1, 0]).tolist()
        noises2 = feature_selection.mutual_info_classif(df_X2, df_y2.values.flatten().reshape(-1, ),
                                                        discrete_features=[1, 0]).tolist()
        noises.append(noises2[0])
        noises_sum = [a + b for a, b in zip(noises, noises_sum)]  # add lists element-wise
    noises_avg = [a / average_over for a in noises_sum]

    df_heatmap = pd.DataFrame({column_name + ' missing values': mis, 'noise': noises_avg},
                              index=[column_w_nan, 'event_type', 'offered_discount'])

    plt.subplots(figsize=(8, 6))
    sns.heatmap(df_heatmap, annot=True)
    plt.savefig("output/sessions_mutual_info_{}.jpg".format(column_name))
    print('saved to output')


def products_mutual_info(df):
    print('{price mutual information}')

    df_a = df.copy()

    df_a.drop(['product_name'], axis=1, inplace=True)
    df_a.drop(['product_id'], axis=1, inplace=True)

    df_a['price'] = np.where((df_a['price'] < 10 ** 4) & (df_a['price'] > 0), 0, df_a['price'])
    df_a['price'] = np.where(df_a['price'] < 0, 1, df_a['price'])
    df_a['price'] = np.where(df_a['price'] >= 10 ** 4, 1, df_a['price'])
    df_a['price'] = df_a['price'].astype(int)

    mi, noise_mi = mutual_info_score_with_noise(df_a, 'category_path', 'price')

    print('out-of-range price and category_path mutual information score: {:.4f}'.format(mi))
    print('mutual information score noise: {:>37.4f}'.format(noise_mi))
