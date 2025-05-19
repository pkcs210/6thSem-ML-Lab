from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict
y_pred = rf_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Random Forest Accuracy:", accuracy)

import numpy as np

class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X[:, self.feature_index] < self.threshold] = -1
        else:
            predictions[X[:, self.feature_index] > self.threshold] = -1
        return predictions

def adaboost(X, y, n_clf=5):
    n_samples, n_features = X.shape
    w = np.full(n_samples, (1 / n_samples))
    models = []
    alphas = []

    for _ in range(n_clf):
        clf = DecisionStump()
        min_error = float('inf')

        # Find best stump
        for feature_i in range(n_features):
            feature_values = np.unique(X[:, feature_i])
            for threshold in feature_values:
                for polarity in [1, -1]:
                    predictions = np.ones(n_samples)
                    if polarity == 1:
                        predictions[X[:, feature_i] < threshold] = -1
                    else:
                        predictions[X[:, feature_i] > threshold] = -1
                    error = np.sum(w[y != predictions])
                    if error < min_error:
                        min_error = error
                        clf.polarity = polarity
                        clf.threshold = threshold
                        clf.feature_index = feature_i

        # Compute alpha
        EPS = 1e-10
        alpha = 0.5 * np.log((1 - min_error + EPS) / (min_error + EPS))
        predictions = clf.predict(X)
        w *= np.exp(-alpha * y * predictions)
        w /= np.sum(w)

        models.append(clf)
        alphas.append(alpha)

    return models, alphas

def predict(X, models, alphas):
    clf_preds = [alpha * clf.predict(X) for clf, alpha in zip(models, alphas)]
    y_pred = np.sign(np.sum(clf_preds, axis=0))
    return y_pred

# Example usage:
# X = np.array([[...], ...])  # shape (n_samples, n_features)
# y = np.array([...])         # shape (n_samples,), labels must be -1 or 1

# models, alphas = adaboost(X, y, n_clf=5)
# y_pred = predict(X, models, alphas)
