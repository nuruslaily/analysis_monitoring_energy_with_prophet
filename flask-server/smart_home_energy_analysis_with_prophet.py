

import numpy as np # linear algebra
import pandas as pd
from pandas import Series # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
sns.set()

from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics

df = pd.read_csv("block_0.csv")

# df.head()

import pandas as pd
import numpy as np
pd.Series(dtype='m8[ns]')
pd.Series(dtype=np.timedelta64(0, 'ns').dtype)
# df.dtypes
# pd(dtype=np.datetime64(0, 'ns').dtype)

# set tstp to index with datetime type
df = df.set_index("tstp")
df.index = df.index.astype("datetime64[ns]")

# set energy consumption data to float type
df = df[df["energy(kWh/hh)"] != "Null"]
df["energy(kWh/hh)"] = df["energy(kWh/hh)"].astype("float64")

# Choose only 1 house by LCLid "MAC000002"
df = df[df["LCLid"] == "MAC000002" ]

# plot energy consumption data with dataframe module
df.plot(y="energy(kWh/hh)", figsize=(12, 4))

train_size = int(0.8 * len(df))
X_train, X_test = df[:train_size].index, df[train_size:].index
y_train, y_test = df[:train_size]["energy(kWh/hh)"].values, df[train_size:]["energy(kWh/hh)"].values

train_df = pd.concat([pd.Series(X_train), pd.Series(y_train)], axis=1, keys=["ds", "y"])
test_df = pd.concat([pd.Series(X_test), pd.Series([0]*len(y_test))], axis=1, keys=["ds", "y"])
answer_df = pd.concat([pd.Series(X_test), pd.Series(y_test)], axis=1, keys=["ds", "y"])

"""## Basic Prediction with Prophet"""

# make model with Prophet by Facebook
model = Prophet()
model.fit(train_df)

forecast = model.predict(test_df)

# forecast.head()

model.plot(forecast)

# with plot_components method, we can visualize the data components
fig2 = model.plot_components(forecast)


# Analysis with cross validation method
# This cell takes some minutes.
# df_cv = cross_validation(model, horizon="60 days")
# df_cv.head()

# With performance_metrics, we can visualize the score
# df_p = performance_metrics(df_cv)
# df_p

# plt.plot(answer_df["ds"], answer_df["y"])
# plt.plot(forecast["ds"], forecast["yhat"])
