import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(iris.data)

colors = ['navy', 'turquoise', 'darkorange']
plt.figure(figsize=(8, 8))
for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
    plt.scatter(X_pca[iris.target == i, 0], X_pca[iris.target == i, 1], color=color, lw=2, label=target_name)
plt.title("PCA of iris dataset")
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.axis([-4, 4, -1.5, 1.5])
plt.show()


pca4d = PCA(n_components=4)
pca4d.fit(iris.data)
X_pca = pca4d.transform(iris.data)

explained_variances = np.zeros(5)
np.cumsum(pca4d.explained_variance_ratio_, out=explained_variances[1:])

fig = plt.figure()
plt.plot(explained_variances)
plt.title('explained variances of Iris PCA')
ax = fig.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()
