from sklearn import datasets, preprocessing
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import seaborn as sns

iris = datasets.load_iris()

ward = AgglomerativeClustering(n_clusters=3)
complete = AgglomerativeClustering(n_clusters=3, linkage='complete')
avg = AgglomerativeClustering(n_clusters=3, linkage='average')


ward_pred = ward.fit_predict(iris.data)
complete_pred = complete.fit_predict(iris.data)
avg_pred = avg.fit_predict(iris.data)

ward_ar_score = adjusted_rand_score(iris.target, ward_pred)
complete_ar_score = adjusted_rand_score(iris.target, complete_pred)
avg_ar_score = adjusted_rand_score(iris.target, avg_pred)

print( "Scores: \nWard:", ward_ar_score,"\nComplete: ", complete_ar_score, "\nAverage: ", avg_ar_score)


# iris.data[:10]
normalized_X = preprocessing.normalize(iris.data)
# normalized_X[:10]

ward_pred = ward.fit_predict(normalized_X)
complete_pred = complete.fit_predict(normalized_X)
avg_pred = avg.fit_predict(normalized_X)

ward_ar_score = adjusted_rand_score(iris.target, ward_pred)
complete_ar_score = adjusted_rand_score(iris.target, complete_pred)
avg_ar_score = adjusted_rand_score(iris.target, avg_pred)

print( "Scores: \nWard:", ward_ar_score,"\nComplete: ", complete_ar_score, "\nAverage: ", avg_ar_score)


linkage_matrix = linkage(normalized_X, 'ward')
plt.figure()
dendrogram(linkage_matrix)
plt.show()

#plt.figure()
sns.clustermap(normalized_X, figsize=(12,18), method='ward', cmap='viridis')
plt.show()
