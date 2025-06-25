import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset from the provided URL
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
df = pd.read_csv(url)

# Define features and target
features = df[['total_bill', 'size']]
target = df['tip']

# print the first few rows of the features and target
print("Features (first few rows):")
print(features.head())
print("\nTarget (first few rows):")
print(target.head())

# split the dataset into training and testing sets
# 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

print("\nTraining data set")
print(X_train.shape)
print("\nTraining data set")
print(X_test.shape)

# Visualize the training data
sns.pairplot(df, x_vars='total_bill', y_vars='tip', aspect=0.8, kind='scatter')
plt.title("Feature vs Target Visualization")
plt.show()