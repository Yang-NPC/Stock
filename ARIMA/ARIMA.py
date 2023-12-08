import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

file_path = '../DataAquire/StockData/MinuteWise/INFY.NS.csv'
data = pd.read_csv(file_path)
data = data[:-1]
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data = data.dropna(subset=['Close'])
close_prices = data['Close'].values.reshape(-1, 1)

# Use 'Close' as the time series
timeseries = close_prices

# Define the order of the ARIMA model
p = 1  # The number of lag observations (lag order)
d = 1  # The number of times that the raw observations are differenced (degree of differencing)
q = 1  # The size of the moving average window (order of moving average)

# Fit the ARIMA model
model = ARIMA(timeseries, order=(p, d, q))
model_fit = model.fit()

# Forecast future values - replace 'steps' with the number of future steps you want to predict
steps = 5
forecast = model_fit.forecast(steps=steps)

# Plot the original data and the forecast
plt.figure(figsize=(10, 5))
plt.plot(timeseries, label='Original')
plt.plot(forecast, label='Forecast', color='red')
plt.title('Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# You can also print out the forecast values
print(forecast)