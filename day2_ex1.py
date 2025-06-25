import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# generate synthetic data
np.random.seed(42)
x = np.random.rand(100, 1) * 100  # 100 samples, single feature
y = 3 * x + np.random.randn(100, 1) * 2  # linear relation with noise

# split the data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# print the coefficients
print("Slope (Coefficient):", model.coef_[0][0])
print("Intercept:", model.intercept_[0])

# finished the simple linear regression model

# Visualize the results
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.title('Linear Regression Results')
plt.xlabel('Feature (x)')
plt.ylabel('Target (y)')
plt.legend()
plt.show()

# Evaluate the performance
# mse is mean squared error which is a measure of the average squared difference between the estimated values and the actual value
mse = mean_squared_error(y_test, y_pred)
# r2_score is the coefficient of determination which indicates how well the model explains the variability of the target variable
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
# If the R^2 score is close to 1, it means the model explains a large portion of the variance in the target variable.
# If the R^2 score is close to 0, it means the model does not explain much of the variance in the target variable.