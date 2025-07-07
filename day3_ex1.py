import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# load the California housing dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# select feature (Median Income) and target (Median House Value)
x = df[['MedInc']]
y = df['MedHouseVal']

# Transform the feature to include polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(x)

# # Fit a polynomial regression model
# model = LinearRegression()
# model.fit(X_poly, y)

# # Make predictions
# y_pred = model.predict(X_poly)

# # Plot actual vs predicted values
# plt.figure(figsize=(10, 6))
# plt.scatter(X, y, color='blue', label='Actual values', alpha=0.5)
# plt.scatter(X, y_pred, color='red', label='Predicted values', alpha=0.5)
# plt.title('Polynomial Regression: Actual vs Predicted')
# plt.xlabel('Median Income')
# plt.ylabel('Median House Value')
# plt.legend()
# plt.show()

# # Evalute the model performance
# mse = mean_squared_error(y, y_pred)
# print(f'Mean Squared Error: {mse:.2f}')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# ridge regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)

# Lasso regression
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X_train, y_train)
lasso_pred = lasso_model.predict(X_test)

# Evaluate the ridge regression model
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)

print(f'Ridge Regression - Mean Squared Error: {ridge_mse:.2f}, R^2 Score: {ridge_r2:.2f}')

# Evaluate the lasso regression model
lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_r2 = r2_score(y_test, lasso_pred)
print(f'Lasso Regression - Mean Squared Error: {lasso_mse:.2f}, R^2 Score: {lasso_r2:.2f}')

# Plotting the predictions
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], y_test, color='blue', label='Actual values', alpha=0.5)
plt.scatter(X_test[:, 0], ridge_pred, color='green', label='Ridge Predicted values', alpha=0.5)
plt.scatter(X_test[:, 0], lasso_pred, color='orange', label='Lasso Predicted values', alpha=0.5)
plt.title('Ridge and Lasso Regression Predictions')
plt.xlabel('Median Income (Transformed)')
plt.ylabel('Median House Value')
plt.legend()
plt.show()