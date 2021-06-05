from sklearn import feature_selection, preprocessing

from utils.json_read import get_jsonl_data
from utils.csv_read import get_csv_data
from preprocess.utils import SEED, check_if_empty, check_if_numeric, check_range, check_timestamp, check_event_type, \
    delete_nulls, fix_null_user_ids, delete_constraints_violations, count_sessions_by_purchase, \
    add_is_buy, session_mutual_info_for_input, check_constraints

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


def sessions_check():
    df_s = pd.DataFrame(get_jsonl_data('data/', 'sessions.jsonl'))
    df_p = get_csv_data('output/', 'products.csv')
    df_u = get_csv_data('output/', 'users.csv')

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
    check_range(df_s, 100001, 120000, 'session_id')

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

    print('\n{offered_discount}')
    check_if_empty(df_s, 'offered_discount')
    check_if_numeric(df_s, 'offered_discount')
    check_range(df_s, 0, 100, 'offered_discount')

    session_mutual_info(df_s)

    print('\n{after checking if missing data is MCAR}')
    print('\nClean missing data')

    df_s = delete_nulls(df_s, 'product_id')
    df_s = fix_null_user_ids(df_s)

    df_s['user_id'] = df_s['user_id'].astype(int)
    check_constraints(df_s, df_u, 'user_id')

    df_s['product_id'] = df_s['product_id'].astype(int)
    check_constraints(df_s, df_p, 'product_id')

    df_s = df_s.drop(['purchase_id'], axis=1)
    df_s = delete_constraints_violations(df_s, df_p, 'product_id')
    df_s = delete_constraints_violations(df_s, df_u, 'user_id')

    df_s = df_s.drop(['user_id'], axis=1)
    df_s = df_s.drop(['product_id'], axis=1)

    check_event_type(df_s)
    count_sessions_by_purchase(df_s)

    # df_s['offered_discount'] = df_s['offered_discount'].astype(float)

    df_s['month'] = pd.DatetimeIndex(df_s['timestamp']).month
    df_s['day'] = pd.DatetimeIndex(df_s['timestamp']).day
    df_s['weekDay'] = pd.DatetimeIndex(df_s['timestamp']).weekday
    df_s['hour'] = pd.DatetimeIndex(df_s['timestamp']).hour

    df_s = df_s.drop(['timestamp'], axis=1)
    df_s = add_is_buy(df_s)

    df_s = df_s[['session_id', 'price', 'offered_discount', 'category_path', 'city', 'month', 'day',
                 'weekDay', 'hour', 'is_buy']]
    df_s.to_csv("output/sessions.csv", sep=';', encoding='utf-8', index=False)

    le = preprocessing.LabelEncoder()
    # df_s['offered_discount'] = le.fit_transform(df_s['offered_discount'].values)
    df_s['category_path'] = le.fit_transform(df_s['category_path'].values)
    df_s['city'] = le.fit_transform(df_s['city'].values)
    df_s['month'] = le.fit_transform(df_s['month'].values)
    df_s['day'] = le.fit_transform(df_s['day'].values)
    df_s['weekDay'] = le.fit_transform(df_s['weekDay'].values)
    df_s['hour'] = le.fit_transform(df_s['hour'].values)

    df_s.to_csv("output/sessions_encoded.csv", sep=';', encoding='utf-8', index=False)

    # with pd.option_context('display.max_columns', None):
    #     print(df_s)

    session_mutual_info_for_input(df_s)


def session_mutual_info(df):
    print('{sessions user_id mutual information scores}')
    display_mutual_info(df, 'user_id', 'product_id')

    print('{sessions product_id mutual information scores}')
    display_mutual_info(df, 'product_id', 'user_id')


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
                                                discrete_features=[0]).tolist()
    mis2 = feature_selection.mutual_info_classif(df_X2, df_y2.values.flatten().reshape(-1, ),
                                                 discrete_features=[0]).tolist()
    mis.append(mis2[0])

    noises_sum = [0, 0, 0]

    np.random.seed(SEED)
    average_over = 100
    for i in range(average_over):
        df_y['checkMCAR'] = np.random.permutation(df_y['checkMCAR'].values)
        df_y2['checkMCAR'] = np.random.permutation(df_y2['checkMCAR'].values)
        noises = feature_selection.mutual_info_classif(df_X, df_y.values.flatten().reshape(-1, ),
                                                       discrete_features=[0]).tolist()
        noises2 = feature_selection.mutual_info_classif(df_X2, df_y2.values.flatten().reshape(-1, ),
                                                        discrete_features=[0]).tolist()
        noises.append(noises2[0])
        noises_sum = [a + b for a, b in zip(noises, noises_sum)]  # add lists element-wise
    noises_avg = [a / average_over for a in noises_sum]

    df_heatmap = pd.DataFrame({column_name + ' missing values': mis, 'noise': noises_avg},
                              index=[column_w_nan, 'event_type', 'offered_discount'])

    plt.subplots(figsize=(8, 6))
    sb.heatmap(df_heatmap, annot=True)
    plt.savefig("output/sessions_mutual_info_{}.jpg".format(column_name))
    print('saved to output')

