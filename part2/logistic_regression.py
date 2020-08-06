import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


def normalize(X, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    learning_rate = 0.1
    W = None

    def init_weights(self, n_features):
        limit = 1 / np.sqrt(n_features)
        self.W = np.random.uniform(-limit, limit, (n_features,))

    def fit(self, X, y, n_iterations=4000):
        self.init_weights(X.shape[1])

        for i in range(n_iterations):
            y_pred = sigmoid(X.dot(self.W))
            self.W -= self.learning_rate * -(y - y_pred).dot(X)

    def predict(self, X):
        y_pred = np.round(sigmoid(X.dot(self.W))).astype(int)
        return y_pred


data = datasets.load_iris()
X = normalize(data.data[data.target != 0])
y = data.target[data.target != 0]
y[y == 1] = 0
y[y == 2] = 1

clf = LogisticRegression()
clf.fit(X, y)
y_pred = clf.predict(X)
