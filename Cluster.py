from BuildDataSet import PointSet
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

pointSet = PointSet()
pointSet.build()
# pointSet.show()
data = pointSet.get_matrix()
estimator = KMeans(n_clusters=pointSet.center_num, max_iter=5)
estimator.fit(data)
labels = estimator.labels_  # 获取聚类标签
centroids = estimator.cluster_centers_  # 获取聚类中心
clusters = []
for i in range(0, pointSet.center_num):
    clusters.append([])
for i in range(0, pointSet.point_num):
    clusters[labels[i]].append(pointSet.point_set[i])
x = []
y = []
for i in range(0, pointSet.center_num):
    for j in clusters[i]:
        x.append(j.x)
        y.append(j.y)
    plt.scatter(x, y, label='Cluster ' + str(i))
    x.clear()
    y.clear()
for i in range(0, pointSet.center_num):
    x.append(centroids[i, 0])
    y.append(centroids[i, 1])
plt.scatter(x, y, label='Centroids')
plt.legend()
plt.show()
