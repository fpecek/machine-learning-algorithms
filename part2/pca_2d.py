import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
sns.set()


rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.axis('equal')
plt.show()


pca2d = PCA(n_components=2)
pca2d.fit(X)
# pca2d.components_, pca2d.explained_variance_, pca2d.explained_variance_ratio_

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->', linewidth=2, shrinkA=0, shrinkB=0, color="0")
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# plot data
plt.figure()
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
for length, vector in zip(pca2d.explained_variance_, pca2d.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca2d.mean_, pca2d.mean_ + v)
plt.axis('equal')
plt.show()


pca1d = PCA(n_components=1)
pca1d.fit(X)
X_pca = pca1d.transform(X)

X_new = pca1d.inverse_transform(X_pca)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
plt.axis('equal')
plt.show()
