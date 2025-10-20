import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1) Load the data — change the path if needed
data = pd.read_csv(r"C:\Users\USER PC\PycharmProjects\PythonProject\Sales.csv")

# Show the first rows so you can confirm columns
print("Preview of data:\n", data.head())

# 2) Convert month names to numbers (if you don't already have numeric months).
# Here we only need a sequence number for each month row:
month_numbers = np.arange(1, len(data['Month']) + 1).reshape(-1, 1)

# 3) Train a Linear Regression model
model = LinearRegression()
# Make sure the target column name is exactly 'Sales' (not 'Sales.csv')
model.fit(month_numbers, data['Sales'])

# 4) Predict sales for the next 2 months (month 13 and 14)
future_months = np.array([[len(month_numbers) + 1], [len(month_numbers) + 2]])
future_sales = model.predict(future_months)
print("Predictions for next months:", future_sales)

# 5) Plot the sales data and prediction line
plt.scatter(month_numbers, data['Sales'], color='red', label='Sales (actual)')
# plot the fitted line across the existing months
plt.plot(month_numbers, model.predict(month_numbers), color='blue', label='Fitted line')
# plot the future predictions as markers (optional)
plt.scatter(future_months, future_sales, color='green', marker='x', label='Predictions')

plt.xlabel('Month (numeric)')
plt.ylabel('Sales')
plt.title('Sales and Linear Regression Prediction')
plt.legend()
plt.show()