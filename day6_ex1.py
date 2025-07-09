from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

# Load Iris Dataset
data = load_iris()
x, y = data.data, data.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Logistic Regression model
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)

# predit using Logistic Regression
y_pred_lr = log_reg.predict(X_test)

# evalalute the logistic regression
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f'Logistic Regression Accuracy: {accuracy_lr:.2f}')

# Evalute k-NN
best_k = 5
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'k-NN Accuracy (k={best_k}): {accuracy_knn:.2f}')

# Detailed comparison
print("\nClassification Report for Logistic Regression:")
print(classification_report(y_test, y_pred_lr))

print("\nClassification Report for k-NN:")
print(classification_report(y_test, y_pred_knn))

# # Experiment with different values of k
# for k in range(1, 11):
#     # Create and train the KNN model
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(X_train, y_train)
    
#     # Make predictions
#     y_pred = knn.predict(X_test)
    
#     # Calculate accuracy
#     accuracy = accuracy_score(y_test, y_pred)
    
#     # Print the results
#     print(f'Accuracy for k={k}: {accuracy:.2f}')
