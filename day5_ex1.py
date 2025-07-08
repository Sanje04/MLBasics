from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier

# load the dataset
data = load_iris()
X, y = data.data, data.target

# Initalize the classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# perform k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

# output the results
print("Cross-validation scores:", cv_scores)
print("Mean accuracy:", cv_scores.mean())