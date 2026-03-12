from sklearn.linear_model import LogisticRegression
import numpy as np

def train_model():
    X = np.array([[1],[2],[3],[4],[5]])
    y = np.array([0,0,0,1,1])
    model = LogisticRegression()
    model.fit(X, y)
    return model

# Test case 1
def test_prediction_class0():
    model = train_model()
    prediction = model.predict([[1]])
    assert prediction[0] == 0

# Test case 2
def test_prediction_class1():
    model = train_model()
    prediction = model.predict([[5]])
    assert prediction[0] == 1

# Test case 3
def test_prediction_boundary():
    model = train_model()
    prediction = model.predict([[4]])
    assert prediction[0] == 1
