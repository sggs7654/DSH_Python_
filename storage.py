from cluster import Cluster
from data_set import PointSet
from sklearn.neighbors import NearestNeighbors
from collections import namedtuple
from math import log2
import numpy as np
import matplotlib.pyplot as plt


class Storage:

    length = 4  # 哈希编码长度(其值应小于len(self.hyperplanes_dict), 该值与neighbors_size正相关)
    neighbors_size = 5  # 质心临近点集容量(其上限为质心数目 - 1), 其值最好大于length
    point_set = None
    cluster = None
    neighbor_indices = None  # 质心临近点索引矩阵, 第一维为质心索引,第二维为临近点索引
    weight = None  # 质心对应簇在数据集中的占比权重所组成的列表,其索引与质心索引一致
    hyperplanes_dict = None  # 保存超平面的字典: key为包含两个质心索引的set, value为包含w,t的命名元组
    hyperplanes_list = None  # 保存筛选后的超平面的列表, 其元素为超平面的wt命名元组
    point_indices_dict = None  # 保存数据经过LSH处理后的索引字典, key为哈希编码组成的元组, value为点索引组成的列表

    def __init__(self):
        self.point_set = PointSet()
        self.cluster = Cluster(self.point_set)
        self.get_centroids_info()  # 计算: 1.质心最近邻索引, 2.各簇占比权重
        self.get_hyperplane_set()  # 计算相邻质心间的超平面参数(w,t)
        self.hyperplane_screening()  # 计算信息熵, 筛选超平面
        self.build_indices()  # 计算数据点的哈希索引(存储索引)

    def get_centroids_info(self):
        # 计算质心临近点索引
        nbrs = NearestNeighbors(n_neighbors=self.neighbors_size + 1, algorithm='auto').fit(self.cluster.centroids)
        self.neighbor_indices = nbrs.kneighbors(self.cluster.centroids, return_distance=False)
        # 离当前质心最近的质心是其本身, 这不是我们所需要的,所以删去
        self.neighbor_indices = np.delete(self.neighbor_indices, 0, axis=1)  # 删除矩阵中的第0列
        # 计算质心对应簇在数据集中的占比权重
        self.weight = []
        for i in range(len(self.cluster.centroids)):
            self.weight.append(0)
        for i in range(len(self.point_set.point_set)):
            centroids_index = self.cluster.labels[i]
            self.weight[centroids_index] = self.weight[centroids_index] + 1

    def get_hyperplane_set(self):
        hyperplane = namedtuple('hyperplane', ['w', 't'])  # 命名元组将作为value保存在超平面字典中
        self.hyperplanes_dict = {}
        for i in range(0, len(self.cluster.centroids)):  # 遍历质心索引
            for j in range(self.neighbors_size):  # 遍历邻近质心索引
                centroids_index1 = i
                centroids_index2 = self.neighbor_indices[i, j]
                # 利用'集合→元组'的方式处理key, 避免索引交换顺序后被重复计算
                key = tuple({centroids_index1, centroids_index2})
                if key not in self.hyperplanes_dict.keys():
                    u1 = self.cluster.centroids[centroids_index1]
                    u2 = self.cluster.centroids[centroids_index2]
                    w = u1 - u2
                    t = np.dot((u1 + u2)/2, w)  # 点积运算
                    hp = hyperplane(w, t)
                    self.hyperplanes_dict[key] = hp

    def hyperplane_screening(self):
        screening_dict = {}  # 用于筛选超平面的字典: key为超平面的信息熵, value为wt命名元组组成的列表
        for hyperplane in self.hyperplanes_dict.values():
            p0 = 0  # p0, p1分别用于累加超平面两侧的质心簇的占比权重
            p1 = 0
            for centroid_index in range(len(self.cluster.centroids)):
                centroid = self.cluster.centroids[centroid_index]
                if np.dot(hyperplane.w, centroid) >= hyperplane.t:
                    p0 = p0 + self.weight[centroid_index]
                else:
                    p1 = p1 + self.weight[centroid_index]
            # print((p0 + p1) == self.point_set.point_num)  # 测试代码: 分布于超平面两侧的点数之和应等于总点数
            entropy = - p0 * log2(p0) - p1 * log2(p1)
            # 考虑到可能存在'两个超平面估算的熵相等'的情况, 所以把wt元组保存在列表中
            if entropy in screening_dict.keys():
                screening_dict[entropy].append(hyperplane)
            else:
                screening_dict[entropy] = [hyperplane]
        keys = list(screening_dict.keys())
        keys.sort(reverse=True)  # 把key降序排列后, 按从高到低的顺序去取length个超平面元组
        self.hyperplanes_list = []
        l_count = 0  # 超平面计数器, 保证筛选后的超平面数量不超过length
        for key in keys:
            # print(key, len(screening_dict[key]), screening_dict[key])  # 测试代码: 打印字典
            for element in screening_dict[key]:
                self.hyperplanes_list.append(element)
                l_count = l_count + 1
                if l_count >= self.length:
                    return

    def build_indices(self):
        self.point_indices_dict = {}  # 保存数据经过LSH处理后的索引字典, key为哈希编码组成的元组, value为点索引组成的列表
        for i in range(self.point_set.point_num):
            point = self.point_set.point_set[i]
            code_list = []  # 点编码列表, 保存点被超平面分割后得到的编码
            for hp in self.hyperplanes_list:
                x = np.array([point.x,point.y])
                if np.dot(hp.w,x) >= hp.t:
                    code_list.append(1)
                else:
                    code_list.append(0)
            key = tuple(code_list)
            if key in self.point_indices_dict.keys():
                self.point_indices_dict[key].append(i)
            else:
                self.point_indices_dict[key] = [i]

    def get_line(self, hp):
        board_min = self.point_set.min_coordinate - self.point_set.radius
        board_max = self.point_set.max_coordinate + self.point_set.radius
        if len(hp.w) > 2:
            raise RuntimeError('数据维度大于2,无法绘图')
        x = []
        y = []
        left_x = board_min
        left_y = (hp.t - hp.w[0] * left_x) / hp.w[1]
        x.append(left_x)
        y.append(left_y)
        right_x = board_max
        right_y = (hp.t - hp.w[0] * right_x) / hp.w[1]
        x.append(right_x)
        y.append(right_y)
        top_y = board_max
        top_x = (hp.t - hp.w[1] * top_y) / hp.w[0]
        x.append(top_x)
        y.append(top_y)
        bottom_y = board_min
        bottom_x = (hp.t - hp.w[1] * bottom_y) / hp.w[0]
        x.append(bottom_x)
        y.append(bottom_y)
        result_x = []
        result_y = []
        for i in range(4):
            if board_min <= x[i] <= board_max and \
                    board_min <= y[i] <= board_max:
                result_x.append(x[i])
                result_y.append(y[i])
        if len(result_x) > 2 or len(result_y) > 2:
            raise RuntimeError('不可能事件:发现超过2个点在画板内')
        return result_x,result_y

    def draw_hyperplane(self):
        for hp in self.hyperplanes_dict.values():
            px,py = self.get_line(hp)
            if max(px) - min(px) > 1000:
                raise RecursionError(hp)
            plt.plot(px, py, color='#C0C0C0', alpha=0.4)
        plt.plot([], [], color='#C0C0C0', label='candidate projection', alpha=0.8)
        x = []
        y = []
        for i in self.point_set.point_set:
            x.append(i.x)
            y.append(i.y)
        plt.scatter(x, y, label='data point')
        plt.legend()
        plt.show()

    def draw_screening(self):
        for hp in self.hyperplanes_dict.values():
            px,py = self.get_line(hp)
            if max(px) - min(px) > 1000:
                raise RecursionError(hp)
            plt.plot(px, py, color='#C0C0C0', alpha=0.2)
        plt.plot([], [], color='#C0C0C0', label='candidate projection', alpha=0.8)
        for hp in self.hyperplanes_list:
            px,py = self.get_line(hp)
            if max(px) - min(px) > 1000:
                raise RecursionError(hp)
            plt.plot(px, py, color='#FF0000', alpha=0.8)
        plt.plot([], [], color='#FF0000', label='high entropy projection', alpha=0.8)
        x = []
        y = []
        for i in self.point_set.point_set:
            x.append(i.x)
            y.append(i.y)
        plt.scatter(x, y, label='data point')
        plt.legend()
        plt.show()
