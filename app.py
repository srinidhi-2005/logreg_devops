import numpy as np
from sklearn.linear_model import LogisticRegression

# training data
X = np.array([[0], [1], [2], [3], [4], [5]])
y = np.array([0, 0, 0, 1, 1, 1]) # binary classification

# training model
model = LogisticRegression()
model.fit(X, y)

# predictions
value = np.array([[4]])
pred = model.predict(value)

print(f"predictions: {pred[0]}")