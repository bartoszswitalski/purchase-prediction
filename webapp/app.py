import flask
import numpy as np
from tensorflow import keras
from datetime import datetime


# load model
model = keras.models.load_model('model')

app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('index.html')

    if flask.request.method == 'POST':

        # fetch data from the form
        price = float(flask.request.form['price'])
        offered_discount = int(flask.request.form['offered_discount'])
        category_path = int(flask.request.form['category_path'])
        city = int(flask.request.form['city'])
        date = datetime.fromisoformat(flask.request.form['date'])
        # process date
        month = date.month
        day = date.day-1
        week_day = date.weekday()
        hour = date.hour

        # prepare data for prediction
        data = np.array([price, offered_discount, category_path, city, month, day, week_day, hour])
        predict_data = list()
        data = data.reshape(1, 8)
        for i in range(8):
            predict_data.append(data[:, i])

        # predict
        prediction = model.predict(predict_data)[0][0][0]

        # render results
        return flask.render_template('index.html',
                                     original_input={'Price': price,
                                                     'Offered discount': offered_discount,
                                                     'Category': category_path,
                                                     'City': city,
                                                     'Month': month,
                                                     'Day': day,
                                                     'Day of the week': week_day,
                                                     'hour': hour},
                                     result=prediction,
                                     )


if __name__ == '__main__':
    app.run()
