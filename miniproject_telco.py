# Task 1: Preform EDA and data preprocessing
# import csv file
link = "https://huggingface.co/datasets/aai510-group1/telco-customer-churn/raw/main/test.csv"
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# load telco customer churn dataset
df = pd.read_csv(link)

# encode categorical variables
label_encoders = LabelEncoder()
df['Churn'] = label_encoders.fit_transform(df['Churn'])

# Define features and target
X = df.drop(columns=['Churn'])
y = df['Churn']

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)

# train K-NN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Evalute models
log_pred = log_model.predict(X_test)
knn_pred = knn_model.predict(X_test)

# print classification reports
print("Logistic Regression Classification Report:")
print(classification_report(y_test, log_pred))
print("k-NN Classification Report:")
print(classification_report(y_test, knn_pred))

# COnfusion matrix for logistic regression
print("Logistic Regression Confusion Matrix: \n", confusion_matrix(y_test, log_pred))

# # inspect the telco dataset
# print(df.info())
# print(df.describe())

# # visualize the churn distribution
# sns.countplot(x='Churn', data=df)
# plt.title('Churn Distribution')
# plt.show()

# # handle missing values
# df.fillna(df.mean(), inplace=True)