# import the nessecary pieces from Flask
import base64
from fbprophet.diagnostics import performance_metrics
from fbprophet.diagnostics import cross_validation
from fbprophet import Prophet
from flask import Flask, render_template, request, jsonify, Response, json
from flask_cors import CORS, cross_origin
import numpy as np  # linear algebra
import pandas as pd
from pandas import Series  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from sympy import true
sns.set()


# Create the a
# pp object that will route our calls
app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'
# @blueprint.after_request # blueprint can also be app~~
# def after_request(response):
#     header = response.headers
#     header['Access-Control-Allow-Origin'] = '*'
# Other headers can be added here if required
# return response

# Add a single endpoint that we can use for testing


@app.route("/data_json", methods=["GET"])
@cross_origin(supports_credentials=True)
def data_json():
    """POST in server"""
    with open('db.json') as f:
        data = json.loads(f.read())
        return ({"data_json": data})


@app.route("/prophet", methods=["GET"])
@cross_origin(supports_credentials=True)
def prophet_pre():
    """POST in server"""
    df = pd.read_csv("block_0.csv")
    pd.Series(dtype='m8[ns]')
    pd.Series(dtype=np.timedelta64(0, 'ns').dtype)
    df = df.set_index("tstp")
    df.index = df.index.astype("datetime64[ns]")

    # set energy consumption data to float type
    df = df[df["energy(kWh/hh)"] != "Null"]
    df["energy(kWh/hh)"] = df["energy(kWh/hh)"].astype("float64")

    # Choose only 1 house by LCLid "MAC00000 2"
    df = df[df["LCLid"] == "MAC000002"]

    # plot energy consumption data with dataframe module
    df.plot(y="energy(kWh/hh)", figsize=(12, 4))
    plt.savefig('plot1.png')

    train_size = int(0.8 * len(df))
    X_train, X_test = df[:train_size].index, df[train_size:].index
    y_train, y_test = df[:train_size]["energy(kWh/hh)"].values, df[train_size:]["energy(kWh/hh)"].values

    train_df = pd.concat(
        [pd.Series(X_train), pd.Series(y_train)], axis=1, keys=["ds", "y"])
    test_df = pd.concat([pd.Series(X_test), pd.Series(
        [0]*len(y_test))], axis=1, keys=["ds", "y"])
    answer_df = pd.concat(
        [pd.Series(X_test), pd.Series(y_test)], axis=1, keys=["ds", "y"])

    # """## Basic Prediction with Prophet"""

    # # make model with Prophet by Facebook
    model = Prophet(yearly_seasonality=True)
    model.fit(train_df)

    forecast = model.predict(test_df)

    model.plot(forecast)
    plt.savefig('forecast.png')

    fig2 = model.plot_components(forecast)
    plt.savefig('forecast_component.png')

    # Analysis with cross validation method
    # This cell takes some minutes.
    cutoffs = pd.to_datetime(['2013-10-12', '2013-11-19'])
    df_cv = cross_validation(model, cutoffs=cutoffs, horizon="60 days")
    df_cv.head()

    # With performance_metrics, we can visualize the score
    df_p = performance_metrics(df_cv)
    df_p

    plt.plot(answer_df['ds'], answer_df['y'])
    plt.plot(forecast['ds'], forecast['yhat'])
    plt.savefig('final.png')

    return 'Berhasil dihitung'
    # return render_template('index.html', shape=df.shape)


# When run from command line, start the server
if __name__ == '__main__':
    app.run(debug=True)
    # return jsonify({"data_json": data_json})
# def member():
#     """POST in server"""
#     return jsonify({"members": ["Member1", "Member2", "Member3"]})

    # parsing data json
    # file_json = os.path.join(app.static_folder, 'db.json')
