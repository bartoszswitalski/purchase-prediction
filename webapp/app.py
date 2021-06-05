import flask
from common import cache
import numpy as np
from tensorflow import keras
from datetime import datetime


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
        price = float(flask.request.form['price'])
        offered_discount = float(flask.request.form['offered_discount'])
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

        # format output strings of floats to 2 digits after decimal point
        price_str = "{:.2f}".format(price)
        offered_discount_str = "{:.2f}".format(offered_discount)

        # prepare log
        categories = {
            0: "Gry i konsole;Gry komputerowe",
            1: "Gry i konsole;Gry na konsole;Gry PlayStation3",
            2: "Gry i konsole;Gry na konsole;Gry Xbox 360",
            3: "Komputery;Drukarki i skanery;Biurowe urządzenia wielofunkcyjne",
            4: "Komputery;Monitory;Monitory LCD",
            5: "Komputery;Tablety i akcesoria;Tablety",
            6: "Sprzęt RTV;Audio;Słuchawki",
            7: "Sprzęt RTV;Przenośne audio i video;Odtwarzacze mp3 i mp4",
            8: "Sprzęt RTV;Video;Odtwarzacze DVD",
            9: "Sprzęt RTV;Video;Telewizory i akcesoria;Anteny RTV",
            10: "Sprzęt RTV;Video;Telewizory i akcesoria;Okulary 3D",
            11: "Telefony i akcesoria;Akcesoria telefoniczne;Zestawy głośnomówiące",
            12: "Telefony i akcesoria;Akcesoria telefoniczne;Zestawy słuchawkowe",
            13: "Telefony i akcesoria;Telefony komórkowe",
            14: "Telefony i akcesoria;Telefony stacjonarne"
        }
        cities = {
            0: "Gdynia",
            1: "Konin",
            2: "Kutno",
            3: "Mielec",
            4: "Police",
            5: "Radom",
            6: "Szczecin",
            7: "Warszawa"
        }
        prediction_log = cache.get('log')
        prediction_log.append({
            'time': f"{datetime.now():%Y-%m-%d %H:%M}",
            'price': price_str,
            'offered_discount': offered_discount_str,
            'category_path': categories[category_path],
            'city': cities[city],
            'date': f"{date:%Y-%m-%d %H:%M}",
            'result': prediction
        })
        cache.set('log', prediction_log)

        # render results
        return flask.render_template('index.html',
                                     original_input={'Price': price_str,
                                                     'Offered discount': offered_discount_str,
                                                     'Category': category_path,
                                                     'City': city,
                                                     'Month': month,
                                                     'Day': day,
                                                     'Day of the week': week_day,
                                                     'Hour': hour},
                                     prediction_log=prediction_log,
                                     result=prediction,
                                     )


if __name__ == '__main__':
    app.run()
