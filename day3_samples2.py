import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Generate synthetic data
np.random.seed(42)
x = np.random.rand(100, 1) * 10  # 100 samples, single feature
y = 3 * x**2 + x*2 + np.random.randn(100, 1) * 5  # quadratic relation with noise

# Transform features to polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(x)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_poly, y, test_size=0.2, random_state=42)

# Ridge regression model
# Ridge regression is a type of linear regression that includes a regularization term to prevent overfitting
# It adds a penalty equal to the square of the magnitude of coefficients
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(x_train, y_train)
ridge_pred = ridge_model.predict(x_test)

# Lasso regression
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(x_train, y_train)
lasso_pred = lasso_model.predict(x_test)

# Evaluate Ridge regression
ridge_mse = mean_squared_error(y_test, ridge_pred)
print("Ridge Mean Squared Error:", ridge_mse)

# Evalute lasso
lasso_mse = mean_squared_error(y_test, lasso_pred)
print("Lasso Mean Squared Error:", lasso_mse)