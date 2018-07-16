from cluster import Cluster
from data_set import PointSet
from sklearn.neighbors import NearestNeighbors
from collections import namedtuple
import numpy as np


class Storage:

    neighbors_size = 2  # 质心临近点集合容量
    point_set = None
    cluster = None
    neighbor_indices = None  # 质心临近点索引
    weight = []  # 质心对应簇在数据集中的占比权重,其索引与质心索引一致
    hyperplanes_dict = {}  # 保存超平面的字典: key为包含两个质心索引的set, value为包含w,t的命名元组

    def __init__(self):
        self.point_set = PointSet()
        self.cluster = Cluster(self.point_set)
        self.get_centroids_info()  # 计算: 1.质心最近邻索引, 2.各簇占比权重

    def get_centroids_info(self):
        # 计算质心临近点索引
        nbrs = NearestNeighbors(n_neighbors=self.neighbors_size + 1, algorithm='auto').fit(self.cluster.centroids)
        self.neighbor_indices = nbrs.kneighbors(self.cluster.centroids, return_distance=False)
        # 离当前质心最近的质心是其本身, 这不是我们所需要的,所以删去
        self.neighbor_indices = np.delete(self.neighbor_indices, 0, axis=1)  # 删除矩阵中的第0列
        # 计算质心对应簇在数据集中的占比权重
        for i in range(0, len(self.cluster.centroids)):
            self.weight.append(0)
        for i in range(0, len(self.point_set.point_set)):
            centroids_index = self.cluster.labels[i]
            self.weight[centroids_index] = self.weight[centroids_index] + 1

    def get_hyperplane_set(self):
        hyperplane = namedtuple('hyperplane',['w','t'])
        for i in range(0,len(self.cluster.centroids)):
            for j in range(0,self.neighbors_size):
                centroids_index1 = i
                centroids_index2 = self.neighbor_indices[i,j]
                u1 = self.cluster.centroids[centroids_index1]
                u2 = self.cluster.centroids[centroids_index2]
                w = u1 - u2
                t = (u1 + u2)/2 * w


# storage = Storage()
# storage.get_hyperplane_set()
# storage.cluster.show()

# 这里先做一个简单的模拟计算, 以判断np向量计算的有效性↓
w = u1 - u2
t = (u1 + u2)/2 * w
