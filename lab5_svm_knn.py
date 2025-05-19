import numpy as np

# Data: [Temp, Humidity, Label]
data = [
    [30, 80, 1], [25, 70, 1], [27, 65, 1], [20, 90, 1],
    [25, 40, 0], [35, 30, 0]
]

def distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

def KNN_predict(test_point, k=3):
    # Compute distances
    dists = sorted(data, key=lambda row: distance(row[:2], test_point))
    # Get labels of k nearest neighbors
    labels = [row[2] for row in dists[:k]]
    # Majority vote
    return 1 if labels.count(1) > labels.count(0) else 0

test_point = [26, 60]
prediction = KNN_predict(test_point)
weather = "Rain" if prediction == 1 else "No Rain"
print(f"Test input: Temperature = {test_point[0]}°C, Humidity = {test_point[1]}%")
print(f"Predicted weather: {weather}")

import numpy as np

# Data: [Temp, Humidity, Label]
data = [
    [30, 80, 1], [25, 70, 1], [27, 65, 1], [20, 90, 1],
    [25, 40, 0], [35, 30, 0]
]

def distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

def KNN_predict(test_point, k=3):
    # Compute distances
    dists = sorted(data, key=lambda row: distance(row[:2], test_point))
    # Get labels of k nearest neighbors
    labels = [row[2] for row in dists[:k]]
    # Majority vote
    return 1 if labels.count(1) > labels.count(0) else 0

test_point = [26, 60]
prediction = KNN_predict(test_point)
weather = "Rain" if prediction == 1 else "No Rain"
print(f"Test input: Temperature = {test_point[0]}°C, Humidity = {test_point[1]}%")
print(f"Predicted weather: {weather}")

