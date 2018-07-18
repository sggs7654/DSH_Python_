import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class Cluster:
    point_set = None
    centroids = None
    labels = None

    def __init__(self, point_set):
        self.point_set = point_set
        data = self.point_set.get_matrix()
        estimator = KMeans(n_clusters=self.point_set.center_num, max_iter=5)  # 初始化聚类器
        estimator.fit(data)  # 拟合模型
        self.labels = estimator.labels_  # 获取聚类标签
        self.centroids = estimator.cluster_centers_  # 获取聚类中心

    def show(self):
        clusters = []
        for i in range(self.point_set.center_num):
            clusters.append([])
        for i in range(self.point_set.point_num):
            clusters[self.labels[i]].append(self.point_set.point_set[i])
        x = []
        y = []
        for i in range(self.point_set.center_num):
            for j in clusters[i]:
                x.append(j.x)
                y.append(j.y)
            plt.scatter(x, y, label='Cluster ' + str(i))
            x.clear()
            y.clear()
        for i in range(self.point_set.center_num):
            x.append(self.centroids[i, 0])
            y.append(self.centroids[i, 1])
            # plt.scatter(self.centroids[i, 0], self.centroids[i, 1], label='Centroids' + str(i))
        plt.scatter(x, y, label='Centroids')
        plt.legend()
        plt.show()
