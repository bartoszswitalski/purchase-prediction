import flask
from common import cache
import numpy as np
import pandas as pd
from tensorflow import keras
from datetime import datetime
from utils.dictionaries import cities, categories


# load model
model = keras.models.load_model('model')

app = flask.Flask(__name__, template_folder='templates')

# init cache to store logs
cache.init_app(app=app, config={"CACHE_TYPE": "filesystem", 'CACHE_DIR': '/tmp'})
cache.set('log', [])


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('index.html',
                                     prediction_log=cache.get('log'))

    if flask.request.method == 'POST':

        # fetch data from the form
        user_id = int(flask.request.form['user_id'])
        product_id = int(flask.request.form['product_id'])
        offered_discount = float(flask.request.form['offered_discount'])
        date = datetime.now()

        # process date
        month = date.month
        day = date.day-1
        week_day = date.weekday()
        hour = date.hour

        # fetch user data to get city
        users_df = pd.read_csv('data/users.csv', sep=';')
        users_dict = users_df.set_index('user_id').to_dict()
        city_str = users_dict['city'][user_id]

        # fetch product data to get price and category
        products_df = pd.read_csv('data/products.csv', sep=';')
        products_dict = products_df.set_index('product_id').to_dict()
        price = products_dict['price'][product_id]
        category_path_str = products_dict['category_path'][product_id]

        # fetch coded values for category and city
        category_path = categories[category_path_str]
        city = cities[city_str]

        # prepare data for prediction
        data = np.array([price, offered_discount, category_path, city, month, day, week_day, hour])
        predict_data = list()
        data = data.reshape(1, 8)
        for i in range(8):
            predict_data.append(data[:, i])

        # predict
        prediction = model.predict(predict_data)[0][0][0]

        # prepare log
        prediction_log = cache.get('log')
        prediction_log.append({
            'position': len(prediction_log)+1,
            'user_id': user_id,
            'product_id': product_id,
            'offered_discount': "{:.2f}".format(offered_discount),
            'price': "{:.2f}".format(price),
            'category_path': category_path_str,
            'city': city_str,
            'date': f"{date:%Y-%m-%d %H:%M}",
            'model': 'A',
            'result': prediction,
            'is_buy': 'Yes' if prediction >= 0.5 else 'No'
        })
        cache.set('log', prediction_log)

        # render results
        return flask.render_template('index.html',
                                     prediction_log=prediction_log,
                                     result=prediction,
                                     )


if __name__ == '__main__':
    app.run()
