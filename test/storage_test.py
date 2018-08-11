from storage import Storage
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt

def get_hyperplane_set_test():
    storage.get_hyperplane_set()
    # 测试1: 确保所有近邻簇都已产生超平面
    index_set = set()
    i = 0
    for indics in storage.neighbor_indices:
        for index in indics:
            index_set.add(tuple({i,index}))
        i = i + 1
    key_set = set()
    for key in storage.hyperplanes_dict.keys():
        key_set.add(key)
    print(key_set == index_set)
    # 测试2: 结果正确性验算(手动指定质心坐标与近邻索引, 验算超平面的w,t)
    centroids = np.array([[1, 1],
                          [2, 1],
                          [1, 2],
                          [2, 2]])
    storage.cluster.centroids = centroids
    storage.neighbor_indices[0,0] = 2
    storage.get_hyperplane_set()
    key = tuple({0, 2})
    u1 = centroids[0]
    u2 = centroids[2]
    w = u1 - u2
    t = np.dot((u1 + u2) / 2, w)
    print(w, t)
    print(storage.hyperplanes_dict[key].w == w)
    print(storage.hyperplanes_dict[key].t == t)


def hyperplane_screening_test():
    storage.hyperplane_screening()
    for i in storage.hyperplanes_list:
        print(i)


def build_indices_test():
    storage.build_indices()
    # print(len(storage.point_indices_dict) <= storage.point_set.point_num)  # 字典容量应小于等于数据总量(小于是因为碰撞)
    # index_count = 0
    # for index_list in storage.point_indices_dict.values():
    #     index_count = index_count + len(index_list)
    # print(index_count == storage.point_set.point_num)  # 索引数目之和应等于数据点数量
    print("哈希编码  索引数量")
    for key in storage.point_indices_dict.keys():
        print(key, len(storage.point_indices_dict[key]))

def get_line_test():
    hyperplane = namedtuple('hyperplane', ['w', 't'])  # 命名元组将作为value保存在超平面字典中
    test_hp = hyperplane(np.array([1.86856672, 74.16929868]), 6478.051434613498)
    x,y = storage.get_line(test_hp)
    minc = storage.point_set.min_coordinate
    maxc = storage.point_set.max_coordinate
    plt.plot([minc, minc], [minc, maxc], color='#C0C0C0')  # left
    plt.plot([minc, maxc], [maxc, maxc], color='#C0C0C0')  # top
    plt.plot([minc, maxc], [minc, minc], color='#C0C0C0')  # bottom
    plt.plot([maxc, maxc], [minc, maxc], color='#C0C0C0')  # right
    plt.plot(x,y)
    plt.show()



storage = Storage()
# get_line_test()
# build_indices_test()
# storage.draw_hyperplane()
# storage.draw_screening()

storage.cluster.show()
# hyperplane_screening_test()
# get_hyperplane_set_test()
# build_indices_test()
# storage.cluster.show()

