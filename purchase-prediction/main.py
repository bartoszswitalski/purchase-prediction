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

if __name__ == '__main__':
    users_check()
    products_check()
    sessions_check()
