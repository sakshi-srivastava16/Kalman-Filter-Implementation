! pip install pykalman
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import statsmodels.api as sm
# crypto currencies of interest - Bitcoin and Ethereum
stock_list = ['BTC-USD', 'ETH-USD']
# extracting the stock price for the last 2 years
data = yf.download(stock_list, start='2023-04-01', end='2025-03-31')
data.head()
data.columns
len(data)
data.describe()
data.corr().style.background_gradient(cmap='coolwarm')
data.isnull().any()
fig, axs =plt.subplots(1,2,figsize=(16, 5),gridspec_kw ={'hspace': 0.2, 'wspace': 0.1})
axs[0].plot(data['Close']['BTC-USD'])
axs[0].set_title('BTC')
axs[1].plot(data['Close']['ETH-USD'])
axs[1].set_title('ETH')
plt.show()
fig, axs =plt.subplots(1,2,figsize=(16, 5),gridspec_kw ={'hspace': 0.2, 'wspace': 0.1})
axs[0].plot(data['Volume']['BTC-USD'])
axs[0].set_title('BTC Volume Trend')
axs[1].plot(data['Volume']['ETH-USD'])
axs[1].set_title('ETH Volume Trend')
plt.show()
from statsmodels.tsa.stattools import adfuller

print("P-value for AD Fuller test for BTC-USD Adj Close {}".format(adfuller( data['Close']['BTC-USD'])[1]))
print("P-value for AD Fuller test for ETH-USD Adj Close {}".format(adfuller( data['Close']['ETH-USD'])[1]))
print("P-value for AD Fuller test for BTC-USD Volume {}".format(adfuller( data['Volume']['BTC-USD'])[1]))
print("P-value for AD Fuller test for ETH-USD Volume {}".format(adfuller( data['Volume']['ETH-USD'])[1]))

#BTC - Stock data
import yfinance as yf
import matplotlib.pyplot as plt

# BTC Closing Price
btc_closing_data = data['Close']['BTC-USD'].loc['2023-04-01':'2025-03-31']
rolling_3d = btc_closing_data.rolling(window=3).mean()
rolling_7d = btc_closing_data.rolling(window=7).mean()
rolling_14d = btc_closing_data.rolling(window=14).mean()
rolling_28d = btc_closing_data.rolling(window=28).mean()

# BTC Volume
btc_volume_data = data['Volume']['BTC-USD'].loc['2024-12-01':'2025-04-01']
volume_rolling_3d = btc_volume_data.rolling(window=3).mean()
volume_rolling_7d = btc_volume_data.rolling(window=7).mean()
volume_rolling_14d = btc_volume_data.rolling(window=14).mean()
volume_rolling_28d = btc_volume_data.rolling(window=28).mean()

# Create subplots (2 rows, 1 column)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 10), sharex=False)

# --- Plot 1: BTC Close Price Rolling Averages ---
axes[0].plot(btc_closing_data.index, btc_closing_data, lw=1.5, alpha=0.8, label='BTC-USD Daily Observation')
axes[0].plot(btc_closing_data.index, rolling_3d, lw=1.5, alpha=0.8, label='Rolling mean - window 3')
axes[0].plot(btc_closing_data.index, rolling_7d, lw=1.5, alpha=0.8, label='Rolling mean - window 7')
axes[0].plot(btc_closing_data.index, rolling_14d, lw=1.5, alpha=0.8, label='Rolling mean - window 14')
axes[0].plot(btc_closing_data.index, rolling_28d, lw=1.5, alpha=0.8, label='Rolling mean - window 28')
axes[0].set_title('BTC-USD Close Price - Observation Window 2024-05-01 : 2024-08-31', fontsize=14)
axes[0].tick_params(labelsize=12)
axes[0].legend(loc='upper left', fontsize=12)

# --- Plot 2: BTC Volume Rolling Averages ---
axes[1].plot(btc_volume_data.index, btc_volume_data, lw=1.5, alpha=0.8, label='BTC-USD Daily Volume')
axes[1].plot(btc_volume_data.index, volume_rolling_3d, lw=1.5, alpha=0.8, label='Rolling mean - window 3')
axes[1].plot(btc_volume_data.index, volume_rolling_7d, lw=1.5, alpha=0.8, label='Rolling mean - window 7')
axes[1].plot(btc_volume_data.index, volume_rolling_14d, lw=1.5, alpha=0.8, label='Rolling mean - window 14')
axes[1].plot(btc_volume_data.index, volume_rolling_28d, lw=1.5, alpha=0.8, label='Rolling mean - window 28')
axes[1].set_title('BTC-USD Volume Shocks - Observation Window 2024-12-01 : 2025-04-01', fontsize=14)
axes[1].tick_params(labelsize=12)
axes[1].legend(loc='upper left', fontsize=12)

# Layout and display
plt.tight_layout()
plt.show()

#ETH - Stock data

import matplotlib.pyplot as plt

# --- ETH Closing Price (April 2023 - March 2025) ---
eth_closing_price = data['Close']['ETH-USD'].loc['2023-04-01':'2025-03-31']
rolling_3d = eth_closing_price.rolling(window=3).mean()
rolling_7d = eth_closing_price.rolling(window=7).mean()
rolling_14d = eth_closing_price.rolling(window=14).mean()
rolling_28d = eth_closing_price.rolling(window=28).mean()

# --- ETH Volume (Dec 2024 - Apr 2025) ---
eth_volume = data['Volume']['ETH-USD'].loc['2024-12-01':'2025-04-01']
volume_3d = eth_volume.rolling(window=3).mean()
volume_7d = eth_volume.rolling(window=7).mean()
volume_14d = eth_volume.rolling(window=14).mean()
volume_28d = eth_volume.rolling(window=28).mean()

# --- Create Subplots ---
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 10), sharex=False)

# --- Plot 1: ETH Closing Price ---
axes[0].plot(eth_closing_price.index, eth_closing_price, lw=1.5, alpha=0.8, label='ETH-USD Daily Observation')
axes[0].plot(eth_closing_price.index, rolling_3d, lw=1.5, alpha=0.8, label='Rolling mean - window 3')
axes[0].plot(eth_closing_price.index, rolling_7d, lw=1.5, alpha=0.8, label='Rolling mean - window 7')
axes[0].plot(eth_closing_price.index, rolling_14d, lw=1.5, alpha=0.8, label='Rolling mean - window 14')
axes[0].plot(eth_closing_price.index, rolling_28d, lw=1.5, alpha=0.8, label='Rolling mean - window 28')
axes[0].set_title('ETH-USD Adj Close Price - Observation Window 2024-05-01 : 2024-08-31', fontsize=14)
axes[0].tick_params(labelsize=12)
axes[0].legend(loc='upper left', fontsize=12)

# --- Plot 2: ETH Volume ---
axes[1].plot(eth_volume.index, eth_volume, lw=1.5, alpha=0.8, label='ETH-USD Daily Volume')
axes[1].plot(eth_volume.index, volume_3d, lw=1.5, alpha=0.8, label='Rolling mean - window 3')
axes[1].plot(eth_volume.index, volume_7d, lw=1.5, alpha=0.8, label='Rolling mean - window 7')
axes[1].plot(eth_volume.index, volume_14d, lw=1.5, alpha=0.8, label='Rolling mean - window 14')
axes[1].plot(eth_volume.index, volume_28d, lw=1.5, alpha=0.8, label='Rolling mean - window 28')
axes[1].set_title('ETH-USD Volume Shocks - Observation Window 2024-12-01 : 2025-04-01', fontsize=14)
axes[1].tick_params(labelsize=12)
axes[1].legend(loc='upper left', fontsize=12)

# --- Final Display ---
plt.tight_layout()
plt.show()

#Understanding the Hyperparameters of Kalman Filters

from pykalman import KalmanFilter
import numpy as np


kalmanFilter = KalmanFilter(transition_matrices = [1],
              observation_matrices = [1],
              initial_state_mean = 0,
              initial_state_covariance = 1,
              observation_covariance=1,
              transition_covariance=.0001)
import matplotlib.pyplot as plt
import pandas as pd

stock_price = data['Close']['BTC-USD']
mean, cov = kalmanFilter.filter(stock_price)
kalman_mean = pd.Series(mean.flatten(), index=stock_price.index)

rolling_3d = stock_price.rolling(window=3).mean()
rolling_7d = stock_price.rolling(window=7).mean()
rolling_14d = stock_price.rolling(window=14).mean()
rolling_28d = stock_price.rolling(window=28).mean()

fig, axs = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

# --- First subplot: Price with smoothing methods
axs[0].plot(stock_price, 'b', lw=1, label='Actual Stock Price')
axs[0].plot(rolling_7d, '-g', lw=1.5, label='7-day Moving Average')
axs[0].plot(rolling_14d, 'm', lw=1, label='14-day Moving Average')
axs[0].plot(rolling_28d, 'y', lw=1, label='28-day Moving Average')
axs[0].plot(kalman_mean, 'r', lw=1, label='Kalman Mean')
axs[0].set_title('BTC-USD Close Price with Kalman and Rolling Means')
axs[0].set_ylabel('Close Price')
axs[0].legend()
axs[0].grid(True)

# --- Second subplot: Differences from smoothing methods
axs[1].plot(stock_price, 'b', lw=1.5, label='Stock Price')
axs[1].plot(kalman_mean, 'm', lw=1.5, label='Kalman Mean')
axs[1].plot(stock_price - kalman_mean, '-g', lw=1.5, label='Diff: Price - Kalman Mean')
axs[1].plot(stock_price - stock_price.mean(), 'r', lw=1.5, label='Diff: Price - Mean')
axs[1].plot(stock_price - rolling_28d, 'y', lw=1.5, label='Diff: Price - 28d MA')
axs[1].plot(stock_price - rolling_3d, 'k', lw=1.5, label='Diff: Price - 3d MA')
axs[1].set_title('Variance of BTC-USD Close Price Compared to Various Estimates')
axs[1].set_xlabel('Day')
axs[1].set_ylabel('Deviation')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import pandas as pd
from pykalman import KalmanFilter

# --- Kalman Filter Setup ---
kalmanFilter = KalmanFilter(
    transition_matrices=[1],
    observation_matrices=[1],
    initial_state_mean=0,
    initial_state_covariance=1,
    observation_covariance=1,
    transition_covariance=0.0001
)

# --- Data and Kalman Estimate ---
stock_price = data['Close']['ETH-USD']
mean, cov = kalmanFilter.filter(stock_price)
kalman_mean = pd.Series(mean.flatten(), index=stock_price.index)

# --- Rolling Averages ---
rolling_3d = stock_price.rolling(window=3).mean()
rolling_7d = stock_price.rolling(window=7).mean()
rolling_14d = stock_price.rolling(window=14).mean()
rolling_28d = stock_price.rolling(window=28).mean()

# --- Combined Plot ---
fig, axs = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

# Plot 1: Price and smoothing estimates
axs[0].plot(stock_price, 'b', lw=1, label='Actual Stock Price')
axs[0].plot(rolling_7d, '-g', lw=1.5, label='7-day Moving Average')
axs[0].plot(rolling_14d, 'm', lw=1, label='14-day Moving Average')
axs[0].plot(rolling_28d, 'y', lw=1, label='28-day Moving Average')
axs[0].plot(kalman_mean, 'r', lw=1, label='Kalman Mean')
axs[0].set_title('ETH-USD Adj Close Price with Kalman Filter and Moving Averages')
axs[0].set_ylabel('Adj Close Price')
axs[0].legend()
axs[0].grid(True)

# Plot 2: Differences from means
axs[1].plot(stock_price, 'b', lw=1.5, label='Stock Price')
axs[1].plot(kalman_mean, 'm', lw=1.5, label='Kalman Mean')
axs[1].plot(stock_price - kalman_mean, '-g', lw=1.5, label='Diff: Price - Kalman Mean')
axs[1].plot(stock_price - stock_price.mean(), 'r', lw=1.5, label='Diff: Price - Mean')
axs[1].plot(stock_price - rolling_28d, 'y', lw=1.5, label='Diff: Price - 28d MA')
axs[1].plot(stock_price - rolling_3d, 'k', lw=1.5, label='Diff: Price - 3d MA')
axs[1].set_title('Deviation of ETH-USD Adj Close from Smoothing Methods')
axs[1].set_xlabel('Day')
axs[1].set_ylabel('Deviation')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

kalmanFilter = KalmanFilter(transition_matrices = [1],
              observation_matrices = [1],
              initial_state_mean = 0,
              initial_state_covariance = 1,
              observation_covariance=1,
              transition_covariance=.01)

# We keep the default parameters but set the transition_covariance to .01 to fit the noise

# Predicting for ETH-USD
stock_price = data['Close']['ETH-USD']

# Kalman Forecast for Next 200 days
mean, cov = kalmanFilter.em(stock_price[:-200], n_iter=10).smooth(stock_price[:-200])
# mean, cov = kalmanFilter.em(stock_price[1], n_iter=10)

next_means = []
next_covs = []
next_mean = mean[-1]
next_cov = cov[-1]
for i in range(200):
  next_mean, next_cov = kalmanFilter.filter_update(next_mean, next_cov, stock_price[-(200 - i)])
  next_means.append(next_mean[0])
  next_covs.append(next_cov[0])


# replacing the forecasted price for the last 10 days
forecasted_price = pd.DataFrame(data=np.concatenate([stock_price[:-200].values, next_means]),
                  index=stock_price.index)

plt.figure(figsize=(12,6))
plt.plot(stock_price[len(stock_price)-200:] ,'b',lw=1.5)
plt.plot(forecasted_price[len(stock_price)-200:] ,'r',lw=1.5)
plt.legend(['Stock Price', 'Kalman Filter Forecasted Price'])
plt.xlabel('Day')
plt.ylabel('Stock Price')
plt.title('ETH-USD - Comparison between Daily Forecasts and Original Stock Price');

from sklearn.metrics import r2_score , mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
import math

_y_stock = stock_price[len(stock_price)-200:] # changed from 10 to 100, after experiment put it back to 10
kalman_forecast_y = forecasted_price[len(stock_price)-200:] # changed from 10 to 100, after experiment put it back to 10
print("R square {}".format(r2_score(_y_stock, kalman_forecast_y)))
print("Mean absolute percentage error {}".format(mean_absolute_percentage_error(_y_stock, kalman_forecast_y)))
print("Root Mean Square Error RMSE {}".format(math.sqrt(mean_squared_error(_y_stock, kalman_forecast_y))))
print("Mean absolute error {}".format(mean_absolute_error(_y_stock, kalman_forecast_y)))

kalmanFilter = KalmanFilter(transition_matrices = [1],
              observation_matrices = [1],
              initial_state_mean = 0,
              initial_state_covariance = 1,
              observation_covariance=1,
              transition_covariance=.01)


# Predicting for ETH-USD
stock_price = data['Close']['ETH-USD']

# Kalman Forecast for Next 100 days
mean, cov = kalmanFilter.em(stock_price[:-100], n_iter=10).smooth(stock_price[:-100]) # changed from 10 to 100, after experiment put it back to 10
# mean, cov = kalmanFilter.em(stock_price[1], n_iter=10)

next_means = []
next_covs = []
next_mean = mean[-1]
next_cov = cov[-1]
for i in range(100):
  next_mean, next_cov = kalmanFilter.filter_update(next_mean, next_cov, stock_price[-(100 - i * 1)]) # changed from 10 to 100, after experiment put it back to 10
  next_means.append(next_mean[0])
  next_covs.append(next_cov[0])

# replacing the forecasted price for the last 100 days
forecasted_price = pd.DataFrame(data=np.concatenate([stock_price[:-100].values, next_means]),
                  index=stock_price.index)

plt.figure(figsize=(12,6))
plt.plot(stock_price[len(stock_price)-100:] ,'b',lw=1.5)
plt.plot(forecasted_price[len(stock_price)-100:] ,'r',lw=1.5)
plt.legend(['Stock Price', 'Kalman Filter Forecasted Price'])
plt.xlabel('Day')
plt.ylabel('Stock Price')
plt.title('Comparison between Daily Forecasts and Original Stock Price for 100 Days');

_y_stock = stock_price[len(stock_price)-100:]
kalman_forecast_y = forecasted_price[len(stock_price)-100:]
print("R square {}".format(r2_score(_y_stock, kalman_forecast_y)))
print("Mean absolute percentage error {}".format(mean_absolute_percentage_error(_y_stock, kalman_forecast_y)))
print("Root Mean Square Error RMSE {}".format(math.sqrt(mean_squared_error(_y_stock, kalman_forecast_y))))
print("Mean absolute error {}".format(mean_absolute_error(_y_stock, kalman_forecast_y)))

kalmanFilter = KalmanFilter(transition_matrices = [1],
              observation_matrices = [1],
              initial_state_mean = 0,
              initial_state_covariance = 1,
              observation_covariance=1,
              transition_covariance=.01)


# Predicting for ETH-USD
stock_price = data['Close']['ETH-USD']

# Kalman Forecast for Next 50 days
mean, cov = kalmanFilter.em(stock_price[:-50], n_iter=10).smooth(stock_price[:-50])
# mean, cov = kalmanFilter.em(stock_price[1], n_iter=10)

next_means = []
next_covs = []
next_mean = mean[-1]
next_cov = cov[-1]
for i in range(50):
  next_mean, next_cov = kalmanFilter.filter_update(next_mean, next_cov, stock_price[-(50 - i * 1)]) # changed from 10 to 100, after experiment put it back to 10
  next_means.append(next_mean[0])
  next_covs.append(next_cov[0])


# replacing the forecasted price for the last 10 days
forecasted_price = pd.DataFrame(data=np.concatenate([stock_price[:-50].values, next_means]),
                  index=stock_price.index)

plt.figure(figsize=(12,6))
plt.plot(stock_price[len(stock_price)-50:] ,'b',lw=1.5)
plt.plot(forecasted_price[len(stock_price)-50:] ,'r',lw=1.5)
plt.legend(['Stock Price', 'Kalman Filter Forecasted Price'])
plt.xlabel('Day')
plt.ylabel('Stock Price')
plt.title('Comparison between Daily Forecasts and Original Stock Price for 50 days');

_y_stock = stock_price[len(stock_price)-50:] # changed from 10 to 100, after experiment put it back to 10
kalman_forecast_y = forecasted_price[len(stock_price)-50:] # changed from 10 to 100, after experiment put it back to 10
print("R square {}".format(r2_score(_y_stock, kalman_forecast_y)))
print("Mean absolute percentage error {}".format(mean_absolute_percentage_error(_y_stock, kalman_forecast_y)))
print("Root Mean Square Error RMSE {}".format(math.sqrt(mean_squared_error(_y_stock, kalman_forecast_y))))
print("Mean absolute error {}".format(mean_absolute_error(_y_stock, kalman_forecast_y)))

kalmanFilter = KalmanFilter(transition_matrices = [1],
              observation_matrices = [1],
              initial_state_mean = 0,
              initial_state_covariance = 1,
              observation_covariance=1,
              transition_covariance=.01)


# Predicting for ETH-USD
stock_price = data['Close']['ETH-USD']

# Kalman Forecast for Next 100 days
mean, cov = kalmanFilter.em(stock_price[:-10], n_iter=10).smooth(stock_price[:-10]) # changed from 10 to 100, after experiment put it back to 10
# mean, cov = kalmanFilter.em(stock_price[1], n_iter=10)

next_means = []
next_covs = []
next_mean = mean[-1]
next_cov = cov[-1]
for i in range(10):
  next_mean, next_cov = kalmanFilter.filter_update(next_mean, next_cov, stock_price[-(10 - i * 1)]) # changed from 10 to 100, after experiment put it back to 10
  next_means.append(next_mean[0])
  next_covs.append(next_cov[0])


# replacing the forecasted price for the last 10 days
forecasted_price = pd.DataFrame(data=np.concatenate([stock_price[:-10].values, next_means]),
                  index=stock_price.index)

plt.figure(figsize=(12,6))
plt.plot(stock_price[len(stock_price)-10:] ,'b',lw=1.5)
plt.plot(forecasted_price[len(stock_price)-10:] ,'r',lw=1.5)
plt.legend(['Stock Price', 'Kalman Filter Forecasted Price'])
plt.xlabel('Day')
plt.ylabel('Stock Price')
plt.title('Comparison between Daily Forecasts and Original Stock Price for 10 days');

_y_stock = stock_price[len(stock_price)-10:] # changed from 10 to 100, after experiment put it back to 10
kalman_forecast_y = forecasted_price[len(stock_price)-10:] # changed from 10 to 100, after experiment put it back to 10
print("R square {}".format(r2_score(_y_stock, kalman_forecast_y)))
print("Mean absolute percentage error {}".format(mean_absolute_percentage_error(_y_stock, kalman_forecast_y)))
print("Root Mean Square Error RMSE {}".format(math.sqrt(mean_squared_error(_y_stock, kalman_forecast_y))))
print("Mean absolute error {}".format(mean_absolute_error(_y_stock, kalman_forecast_y)))

btc_data = {'Open': data['Open']['BTC-USD'],
            'Close': data['Close']['BTC-USD'],
            'High': data['High']['BTC-USD'],
            'Low': data['Low']['BTC-USD'],
            'Volume': data['Low']['BTC-USD'],
            }

eth_data = {'Open': data['Open']['ETH-USD'],
            'Close': data['Close']['ETH-USD'],
            'High': data['High']['ETH-USD'],
            'Low': data['Low']['ETH-USD'],
            'Volume': data['Low']['ETH-USD'],
            }

btc_df = pd.DataFrame(btc_data, columns = ['Open', 'Close', 'High', 'Low', 'Volume'])
eth_df = pd.DataFrame(eth_data, columns = ['Open', 'Close', 'High', 'Low', 'Volume'])

eth_df.head()
eth_df.describe()

btc_df.describe()

eth_df['Target'] = eth_df['Close'].shift(-1)
eth_df.dropna(inplace=True)
eth_df.head()
eth_df.tail()

forecast_set = eth_df[-200:] # validation set, we will keep the last 200 indices for forecast
training_set = eth_df[:-200] # The other we will be using for training

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

X = training_set.drop('Target', axis=1)
y = training_set['Target']


# calling a standard scaler
standard_scaler = preprocessing.StandardScaler().fit(X)
X_scaled = standard_scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)


lr = LinearRegression(n_jobs=-1)
lr.fit(X_train, y_train)

score = lr.score(X_test, y_test)
print('The linear regression confidence is {}'.format(score))
prediction = lr.predict(X_test)
from sklearn.metrics import r2_score , mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
import math

print("R square {}".format(r2_score(y_test, prediction)))
print("Mean absolute percentage error {}".format(mean_absolute_percentage_error(y_test, prediction)))
print("Root Mean Square Error RMSE {}".format(math.sqrt(mean_squared_error(y_test, prediction))))
print("Mean absolute error {}".format(mean_absolute_error(y_test, prediction)))

plt.figure(figsize=(12,6))
plt.plot(y_test ,'b',lw=1.5)
plt.plot(pd.DataFrame(data=prediction, index=y_test.index) ,'r',lw=1.5)
plt.legend(['Actual Stock Price', 'Linear Regression - Predicted Stock Price'])
plt.xlabel('Day')
plt.ylabel('Stock Price')
plt.title('Comparison between Daily Forecasts and Original Stock Price - Test Set');

forecast_X = forecast_set.drop('Target', axis=1)
forecast_y = forecast_set['Target']
scaled_forecast_X = standard_scaler.transform(forecast_X) # using the standard scaler from the training

prediction = lr.predict(scaled_forecast_X[:200])
from sklearn.metrics import r2_score , mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
import math

print("R square {}".format(r2_score(forecast_y[:200], prediction)))
print("Mean absolute percentage error {}".format(mean_absolute_percentage_error(forecast_y[:200], prediction)))
print("Root Mean Square Error RMSE {}".format(math.sqrt(mean_squared_error(forecast_y[:200], prediction))))
print("Mean absolute error {}".format(mean_absolute_error(forecast_y[:200], prediction)))

plt.figure(figsize=(12,6))
plt.plot(forecast_y[:200] ,'b',lw=1.5)
plt.plot(pd.DataFrame(data=prediction, index=forecast_y[:200].index) ,'r',lw=1.5)
plt.legend(['Actual Stock Price', 'Linear Regression - Predicted Stock Price'])
plt.xlabel('Day')
plt.ylabel('Stock Price')
plt.title('Comparison between Daily Forecasts for 200 Days and Original Stock Price - Forecast Set');

prediction = lr.predict(scaled_forecast_X[:100])
from sklearn.metrics import r2_score , mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
import math

print("R square {}".format(r2_score(forecast_y[:100], prediction)))
print("Mean absolute percentage error {}".format(mean_absolute_percentage_error(forecast_y[:100], prediction)))
print("Root Mean Square Error RMSE {}".format(math.sqrt(mean_squared_error(forecast_y[:100], prediction))))
print("Mean absolute error {}".format(mean_absolute_error(forecast_y[:100], prediction)))

plt.figure(figsize=(12,6))
plt.plot(forecast_y[:100] ,'b',lw=1.5)
plt.plot(pd.DataFrame(data=prediction, index=forecast_y[:100].index) ,'r',lw=1.5)
plt.legend(['Actual Stock Price', 'Linear Regression - Predicted Stock Price'])
plt.xlabel('Day')
plt.ylabel('Stock Price')
plt.title('Comparison between Daily Forecasts and Original Stock Price - Forecast Set for 100 days');

prediction = lr.predict(scaled_forecast_X[:50])
from sklearn.metrics import r2_score , mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
import math

print("R square {}".format(r2_score(forecast_y[:50], prediction)))
print("Mean absolute percentage error {}".format(mean_absolute_percentage_error(forecast_y[:50], prediction)))
print("Root Mean Square Error RMSE {}".format(math.sqrt(mean_squared_error(forecast_y[:50], prediction))))
print("Mean absolute error {}".format(mean_absolute_error(forecast_y[:50], prediction)))

plt.figure(figsize=(12,6))
plt.plot(forecast_y[:50] ,'b',lw=1.5)
plt.plot(pd.DataFrame(data=prediction, index=forecast_y[:50].index) ,'r',lw=1.5)
plt.legend(['Actual Stock Price', 'Linear Regression - Predicted Stock Price'])
plt.xlabel('Day')
plt.ylabel('Stock Price')
plt.title('Comparison between Daily Forecasts and Original Stock Price - Forecast Set for 50 days');

prediction = lr.predict(scaled_forecast_X[:10])
from sklearn.metrics import r2_score , mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
import math

print("R square {}".format(r2_score(forecast_y[:10], prediction)))
print("Mean absolute percentage error {}".format(mean_absolute_percentage_error(forecast_y[:10], prediction)))
print("Root Mean Square Error RMSE {}".format(math.sqrt(mean_squared_error(forecast_y[:10], prediction))))
print("Mean absolute error {}".format(mean_absolute_error(forecast_y[:10], prediction)))

plt.figure(figsize=(12,6))
plt.plot(forecast_y[:10] ,'b',lw=1.5)
plt.plot(pd.DataFrame(data=prediction, index=forecast_y[:10].index) ,'r',lw=1.5)
plt.legend(['Actual Stock Price', 'Linear Regression - Predicted Stock Price'])
plt.xlabel('Day')
plt.ylabel('Stock Price')
plt.title('Comparison between Daily Forecasts and Original Stock Price - Forecast Set for 10 days');

