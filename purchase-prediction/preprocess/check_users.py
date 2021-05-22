"""
    Name: check_users.py
    Purpose: users.jsonl data validity check

    @author Bartosz Świtalski, Piotr Frątczak

    Warsaw University of Technology
    Faculty of Electronics and Information Technology
"""
from preprocess.json_read import get_jsonl_data
from preprocess.utils import check_if_empty, check_if_unique_values, check_if_numeric, check_range, plot_histogram

import pandas as pd


def users_check():
    df = pd.DataFrame(get_jsonl_data('data/', 'users.jsonl'))
    """
    Check if:
    1) id is an integer (and is not null)
    2) id is unique
    3) id is in good range of values
    
    4) city is not null
    5) city distribution
    """
    print('###### users.jsonl check ######')

    print('{user_id}')
    check_if_empty(df, 'user_id')
    check_if_unique_values(df, 'user_id')
    check_if_numeric(df, 'user_id')
    check_range(df, 102, 301, 'user_id')

    print('{city}')
    check_if_empty(df, 'city')
    plot_histogram(df, 'city')

    df = df.drop(['name', 'street'], axis=1)
    df.to_csv("output/users.csv", sep=';', encoding='utf-8', index=False)
