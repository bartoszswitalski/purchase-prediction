"""
    Name: check_products.py
    Purpose: products.jsonl data validity check

    @author Bartosz Świtalski, Piotr Frątczak

    Warsaw University of Technology
    Faculty of Electronics and Information Technology
"""
import random

from sklearn import preprocessing, feature_selection

from preprocess.json_read import get_jsonl_data

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from preprocess.utils import SEED, check_if_empty, check_if_unique_values, check_if_numeric, \
    check_range, check_if_float, check_if_string, check_boxplot, delete_invalid_values


def products_check():
    df = pd.DataFrame(get_jsonl_data('data/', 'products.jsonl'))
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
    check_if_unique_values(df, 'product_id')
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
    check_boxplot(df, 'price')

    products_mutual_info(df)

    df = delete_invalid_values(df, 'price', 0, 10 ** 4)
    df = df.drop(['product_name'], axis=1)
    df.to_csv("output/products.csv", sep=';', encoding='utf-8', index=False)


def products_mutual_info(df):
    print('\n{price mutual information}')

    df_a = df.copy()

    df_a.drop(['product_name'], axis=1, inplace=True)
    df_a.drop(['product_id'], axis=1, inplace=True)

    df_a['price'] = np.where((df_a['price'] < 10 ** 4) & (df_a['price'] > 0), 0, df_a['price'])
    df_a['price'] = np.where(df_a['price'] < 0, 1, df_a['price'])
    df_a['price'] = np.where(df_a['price'] >= 10 ** 4, 1, df_a['price'])
    df_a['price'] = df_a['price'].astype(int)

    le = preprocessing.LabelEncoder()
    df_a['category_path'] = le.fit_transform(df_a['category_path'].values)
    df_a['dummy'] = random.randint(0, 1)

    df_aux = df_a[['category_path', 'dummy']].copy()

    mis = feature_selection.mutual_info_classif(df_aux, df_a['price'].values.flatten().reshape(-1, ),
                                                discrete_features=[0]).tolist()
    noises_sum = [0, 0]

    np.random.seed(SEED)
    average_over = 100
    for i in range(average_over):
        df_a['price'] = np.random.permutation(df_a['price'].values)
        noises = feature_selection.mutual_info_classif(df_aux, df_a['price'].values.flatten().reshape(-1, ),
                                                       discrete_features=[0]).tolist()

        noises_sum = [a + b for a, b in zip(noises, noises_sum)]  # add lists element-wise
    noises_avg = [a / average_over for a in noises_sum]

    df_heatmap = pd.DataFrame({'price invalid values': mis, 'noise': noises_avg},
                              index=['category_path', 'dummy'])

    plt.subplots(figsize=(8, 6))
    sb.heatmap(df_heatmap, annot=True)
    plt.savefig("output/products_mutual_info_{}.jpg".format('price'))
    print('saved to output')