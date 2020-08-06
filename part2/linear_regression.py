import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt


class LinearRegression:
    W = None
    Ws = []
    training_errors = []

    def init_weights(self, n_features):
        limit = 1 / np.sqrt(n_features)
        self.W = np.random.uniform(-limit, limit, (n_features, ))

    def fit(self, X, y, n_iterations=100, learning_rate=0.01):
        X = np.insert(X, 0, 1, axis=1)
        self.training_errors = []

        self.init_weights(X.shape[1])

        # do gradient descent for n_iterations
        for i in range(n_iterations):
            y_pred = X.dot(self.W)
            
            # calculate l2 loss
            mse = np.mean(0.5 * (y - y_pred)**2)
            self.training_errors.append(mse)
            self.Ws.append(self.W.tolist())

            # gradient of l2 loss wrt w
            grad_W = -(y - y_pred).dot(X)

            # update weights
            self.W -= learning_rate * grad_W

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.W)
        return y_pred


def loss_fn(W, X, y):
    return np.mean(0.5 * (y - X.dot(W))**2)


bias = 10
X, y, coef = make_regression(n_samples=100, n_features=1, noise=30, bias=bias, coef=True)

lc = LinearRegression()
lc.fit(X, y)

print(f'true linear model: {bias} + {coef}*x')
print(f'learned linear model: {lc.W[0]} + {lc.W[1]}*x')


# plot loss surface
# https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
sz = max(np.abs(bias), np.abs(coef))
w1 = np.linspace(bias-2*sz, bias+2*sz, 30)
w2 = np.linspace(coef-2*sz, coef+2*sz, 30)
W1, W2 = np.meshgrid(w1, w2)
WW = np.meshgrid(w1, w2)
XX = np.insert(X, 0, 1, axis=1)

z = np.mean((y[:,np.newaxis] - XX.dot(np.stack(WW).reshape(2,30*30)))**2, axis=0).reshape(30,30)

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection="3d")
ax.plot_surface(W1, W2, z, rstride=1, cstride=1, cmap='terrain', edgecolor=None, alpha=0.7)
ax.plot(np.array(lc.Ws)[:,0], np.array(lc.Ws)[:,1], np.zeros_like(np.array(lc.Ws)).shape[0])
plt.show()


# plot loss surface and proxy loss surface
XX2 = XX[:2]
y2 = y[:2]
z2 = np.mean((y2[:,np.newaxis] - XX2.dot(np.stack(WW).reshape(2,30*30)))**2, axis=0).reshape(30,30)

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection="3d")
ax.plot_surface(W1, W2, z, rstride=1, cstride=1, cmap='terrain', edgecolor=None, alpha=0.7)
ax.plot_surface(W1, W2, z2, rstride=1, cstride=1, cmap='viridis', edgecolor=None, alpha=0.7)
plt.show()

print('min of true function',np.unravel_index(z.argmin(), z.shape))
print('min of proxy function', np.unravel_index(z2.argmin(), z2.shape))
