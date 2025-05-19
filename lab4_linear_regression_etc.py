
import numpy as np
import pandas as pd

# Sample data
data = {
    "Temp": [30, 28, 25, 27, 32],
    "Humidity": [70, 65, 80, 85, 75],
    "Windspeed": [5, 10, 12, 8, 7],
    "Rain": [1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Features and target
X = df[["Temp", "Humidity", "Windspeed"]].values
y = df["Rain"].values

# Feature scaling (Standardization)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Add bias (intercept) term
X = np.c_[np.ones(X.shape[0]), X]

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function
def compute_cost(X, y, weights):
    m = len(y)
    h = sigmoid(np.dot(X, weights))
    cost = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

# Gradient Descent
def gradient_descent(X, y, weights, alpha, iterations):
    m = len(y)
    cost_history = []
    for i in range(iterations):
        h = sigmoid(np.dot(X, weights))
        gradient = np.dot(X.T, (h - y)) / m
        weights -= alpha * gradient
        cost_history.append(compute_cost(X, y, weights))
    return weights, cost_history

# Initialize weights, learning rate, and iterations
weights = np.zeros(X.shape[1])
alpha = 0.1
iterations = 1000

# Train model
weights, cost_history = gradient_descent(X, y, weights, alpha, iterations)
print("Final parameters:", weights)

# Prediction function
def predict(X, weights):
    return sigmoid(np.dot(X, weights)) >= 0.5

# Make predictions
predictions = predict(X, weights).astype(int)
print("Predicted labels:", predictions)

# Calculate accuracy
accuracy = np.mean(predictions == y)
print("Accuracy:", accuracy * 100, "%")


import numpy as np
import pandas as pd

# Example data
data = {
    "Area": [2104, 1600, 2400, 1416, 3000],
    "Bedrooms": [3, 3, 3, 2, 4],
    "Price": [399900, 329900, 369000, 232000, 539900]
}
df = pd.DataFrame(data)

X = df[["Area", "Bedrooms"]].values
y = df["Price"].values

# Feature scaling
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
X = np.c_[np.ones(X.shape[0]), X]  # Add bias

theta = np.zeros(X.shape[1])
alpha = 0.01
iterations = 1000

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum((predictions - y) ** 2)
    return cost

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []
    for i in range(iterations):
        predictions = X.dot(theta)
        gradient = (1/m) * X.T.dot(predictions - y)
        theta -= alpha * gradient
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)
print("Final parameters:", theta)



import numpy as np
import pandas as pd

# Sample data
data = {
    "Temp": [30, 28, 25, 27, 32],
    "Humidity": [70, 65, 80, 85, 75],
    "Windspeed": [5, 10, 12, 8, 7],
    "Rain": [1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Features and target
X = df[["Temp", "Humidity", "Windspeed"]].values
y = df["Rain"].values

# Feature scaling (Standardization)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Add bias (intercept) term
X = np.c_[np.ones(X.shape[0]), X]

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function
def compute_cost(X, y, weights):
    m = len(y)
    h = sigmoid(np.dot(X, weights))
    cost = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

# Gradient Descent
def gradient_descent(X, y, weights, alpha, iterations):
    m = len(y)
    cost_history = []
    for i in range(iterations):
        h = sigmoid(np.dot(X, weights))
        gradient = np.dot(X.T, (h - y)) / m
        weights -= alpha * gradient
        cost_history.append(compute_cost(X, y, weights))
    return weights, cost_history

# Initialize weights, learning rate, and iterations
weights = np.zeros(X.shape[1])
alpha = 0.1
iterations = 1000

# Train model
weights, cost_history = gradient_descent(X, y, weights, alpha, iterations)
print("Final parameters:", weights)

# Prediction function
def predict(X, weights):
    return sigmoid(np.dot(X, weights)) >= 0.5

# Make predictions
predictions = predict(X, weights).astype(int)
print("Predicted labels:", predictions)

# Calculate accuracy
accuracy = np.mean(predictions == y)
print("Accuracy:", accuracy * 100, "%")
