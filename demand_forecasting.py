
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load data
data = pd.read_csv("sample_sales_data.csv")
data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month + 12 * (data['Date'].dt.year - data['Date'].dt.year.min())

# Prepare features and target
X = data[['Month']]
y = data['Sales']

# Fit the model
model = LinearRegression()
model.fit(X, y)

# Forecast next 6 months
future_months = pd.DataFrame({'Month': range(X['Month'].max() + 1, X['Month'].max() + 7)})
future_sales = model.predict(future_months)

# Plot
plt.plot(data['Date'], y, label='Historical Sales')
future_dates = pd.date_range(start=data['Date'].max() + pd.DateOffset(months=1), periods=6, freq='M')
plt.plot(future_dates, future_sales, label='Forecasted Sales', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Demand Forecasting')
plt.legend()
plt.tight_layout()
plt.show()
