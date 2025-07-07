import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
x = np.random.rand(100, 1) * 100  # 100 samples, single feature
y = 3 * x**2 + x*2 + np.random.randn(100, 1) * 5  # quadratic relation with noise

# Transform features to polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(x)

# fit the polynomial regression model
model = LinearRegression()
model.fit(x_poly, y)
y_pred = model.predict(x_poly)

# plot the results
plt.scatter(x, y, color='blue', label='Actual')
plt.scatter(x, y_pred, color='red', label='Predicted')
plt.title('Polynomial Regression Results')
plt.xlabel('Feature (x)')
plt.ylabel('Target (y)')
plt.legend()
plt.show()

# evaluate the model
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error:", mse)
