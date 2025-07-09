# Task 1: Preform EDA and data preprocessing
# import csv file
link = "https://raw.githubusercontent.com/KimathiNewton/Telco-Customer-Churn/refs/heads/master/Datasets/telco_churn.csv"
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = fetch_california_housing(as_frame=True)
# Display the first few rows of the dataset
df = data.frame

# Define the features and target
X = df[['MedInc', 'HouseAge', 'AveRooms']]
y = df['MedHouseVal']

# # Inspect the data
# print(df.info())
# print(df.describe())

# # Visualize relationships
# sns.pairplot(df, vars=['MedInc', 'HouseAge', 'AveRooms', 'MedHouseVal'])
# plt.show()

# # check for missing values
# print("Missing values in each column:")
# print(df.isnull().sum())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# evalute performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Linear Regression MSE:", mse)