import flask
import csv
import pytz
import hashlib
import numpy as np
import pandas as pd
from common import cache
from tensorflow import keras
from datetime import datetime
from utils.dictionaries import cities, categories


# load models
modelA = keras.models.load_model('model/modelA')
modelB = keras.models.load_model('model/modelB')

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
        user_str = flask.request.form['user_id']
        user_id = int(user_str)
        product_id = int(flask.request.form['product_id'])
        offered_discount = float(flask.request.form['offered_discount'])

        # get time
        cet = pytz.timezone('Europe/Warsaw')
        date = datetime.now(cet)

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

        # choose model based on user's id and run prediction
        if int(hashlib.sha256(user_str.encode('utf-8')).hexdigest(), 16) % 2 == 0:
            prediction = modelA.predict(predict_data)[0][0][0]
            model_str = 'A'
        else:
            prediction = modelB.predict(predict_data)[0][0]
            model_str = 'B'

        # log which model was chosen
        print('User ID: {} - Model: {}'.format(user_str, model_str))

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
            'model': model_str,
            'result': prediction,
            'is_buy': 'Yes' if prediction >= 0.5 else 'No'
        })
        cache.set('log', prediction_log)

        # render results
        return flask.render_template('index.html',
                                     prediction_log=prediction_log,
                                     result=prediction,
                                     )


@app.route('/downloads/<path:filename>', methods=['GET'])
def download(filename):
    prediction_log = cache.get('log')
    if len(prediction_log) == 0:
        return main()
    keys = prediction_log[0].keys()
    with open(filename, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys, delimiter=',')
        dict_writer.writeheader()
        dict_writer.writerows(prediction_log)

    return flask.send_file(filename, as_attachment=True)


if __name__ == '__main__':
    app.run()
