import numpy as np
from data_set import Point


class RPTree:
    point_set = None
    c = 1
    labels_ = None
    cluster_centers_ = None

    def __init__(self, point_set):  # 初始化数据集, 设定参数c
        self.point_set = point_set

    # 对数据集执行rp树分割, 返回数据标签和聚类中心
    def fit(self):
        ...

    # 递归函数, 其输入为数据索引集, 输出为rp树各叶子结点的数据索引集
    def make_tree(self, indices):
        delta = self.get_delta(indices)  # 点云直径
        vec_sum = np.array([0, 0])  # 累加数据向量, 用于求平均
        vec_list = []  # 存放数据向量, 用于计算平均点距
        for i in indices:  # 把数据点转换成numpy向量
            p = self.point_set.point_set[i]
            vec = np.array([p.x, p.y])
            vec_sum = vec_sum + vec
            vec_list.append(vec)
        mean_vec = vec_sum / len(indices)
        accumulator = 0
        for v in vec_list:
            temp = v - mean_vec  # temp就是公式中的x - mean(S)
            temp = temp.dot(temp)  # 平方
            accumulator += temp
        average_delta_square = 2 * accumulator / len(indices)
        # return average_delta_square  # 供平均点距测试函数使用, 测试时注释掉后续代码
        if delta * delta <= self.c * average_delta_square:
            ...


    # 计算点云直径, 其输入为数据索引集, 输出为直径或tuple(直径, 端点a, 端点b)
    def get_delta(self, indices, return_indices=False):
        r_old = 0  # 用作比较的r,初始化
        temp_point_set = []
        for index in indices:  # temp_point_set为indices指向的点组成的集合
            temp_point_set.append(self.point_set.point_set[index])
        np.random.seed()
        p_index = np.random.randint(low=0, high=len(temp_point_set) - 1)  # 从点集中随机选一个点
        p = temp_point_set[p_index]
        while True:
            q_index = self.get_furthest_point(p_index, temp_point_set)  # 找到点集中离p最远的点
            q = temp_point_set[q_index]
            q_prime_index = self.get_furthest_point(q_index, temp_point_set)  # 找到点集中离q最远的点
            q_prime = temp_point_set[q_prime_index]
            rho = p.dist(q)
            r = q.dist(q_prime)
            temp_point_set.remove(p)
            temp_point_set.remove(q)
            if r <= r_old or len(temp_point_set) <= 1:  # 如果结果无改善或点集容量不足,则停止,否则继续
                if return_indices:
                    return r, q, q_prime
                else:
                    return r
            r_old = r
            p_vec = np.array([p.x, p.y])
            q_vec = np.array([q.x, q.y])
            # q_prime_vec = np.array([q_prime.x, q_prime.y])
            p_prime_vec = q_vec + (r/rho)*(p_vec - q_vec)      # paper method
            p_vec = 0.5 * (p_prime_vec + q_vec)                # paper method
            # p_vec = 0.5 * (q_prime_vec + q_vec)              # my method(not good)
            p = Point(p_vec[0], p_vec[1])
            temp_point_set.append(p)
            p_index = len(temp_point_set) - 1

    # 找到与目标点距离最远的点, 输入为数据索引, 输出为数据索引, 复杂度为O(n)
    def get_furthest_point(self, index, point_set):
        max_dist = 0
        result_index = None
        for i in range(len(point_set)):
            dist = point_set[index].dist(point_set[i])
            if dist > max_dist:
                max_dist = dist
                result_index = i
        return result_index
