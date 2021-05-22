"""
    Name:
    Purpose:

    @author Bartosz Świtalski, Piotr Frątczak

    Warsaw University of Technology
    Faculty of Electronics and Information Technology
"""
from check_users import users_check
from check_products import products_check
from check_sessions import sessions_check
from preprocess.aggregate import get_aggregated_sessions

if __name__ == '__main__':
    # users_check()
    # products_check()
    # sessions_check()
    get_aggregated_sessions()
