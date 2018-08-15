import matplotlib.pyplot as plt
import numpy as np
from data_set import Point
from RP_Tree.rptree import RPTree


def divide_test_2(point_set):
    # indices = range(3)
    # point_set.point_num = 3
    # point_set.point_set.clear()
    # point_set.point_set.append(Point(0, 0))
    # point_set.point_set.append(Point(10, 0))
    # point_set.point_set.append(Point(20, 0))
    # point_set.show()
    indices = range(point_set.point_num)
    rp_tree = RPTree(point_set)
    p1, p2 = rp_tree.divide(indices)
    print(len(p1) + len(p2) == point_set.point_num)
    # print(p1)
    # print(p2)
    x = []
    y = []
    for i in p1:
        x.append(point_set.point_set[i].x)
        y.append(point_set.point_set[i].y)
    plt.scatter(x, y, label = 'p1')
    x.clear()
    y.clear()
    for i in p2:
        x.append(point_set.point_set[i].x)
        y.append(point_set.point_set[i].y)
    plt.scatter(x, y, label = 'p2')
    plt.legend()
    plt.show()


def divide_test_1(point_set):  # 需要注释掉divide函数中'return平均点距'后面的代码
    indices = range(3)
    point_set.point_set.clear()
    point_set.point_num = 3
    point_set.point_set.append(Point(0, 0))
    point_set.point_set.append(Point(0, 3))
    point_set.point_set.append(Point(4, 0))
    # point_set.show()
    rp_tree = RPTree(point_set)
    average_delta_aquare = rp_tree.divide(indices)
    print(average_delta_aquare)
    right_answer = (3*3 + 4*4 + 5*5)*2 / (3*3)
    print(right_answer)

def get_delta_test(point_set, rp_tree):
    r,p1,p2 = rp_tree.get_delta(range(point_set.point_num),return_indices=True)
    print(r)
    x = []
    y = []
    for i in point_set.point_set:
        x.append(i.x)
        y.append(i.y)
    plt.scatter(x, y)
    plt.plot([p1.x, p2.x], [p1.y, p2.y])
    plt.show()


def get_furthest_point_test(point_set, rp_tree):
    x = []
    y = []
    for i in point_set.point_set:
        x.append(i.x)
        y.append(i.y)
    plt.scatter(x, y)
    np.random.seed()
    target_point_index = np.random.randint(low=0, high=point_set.point_num - 1)
    target_point = point_set.point_set[target_point_index]
    furthest_point_index = rp_tree.get_furthest_point(target_point_index)
    furthest_point = point_set.point_set[furthest_point_index]
    plt.scatter(target_point.x, target_point.y, label='target')
    plt.scatter(furthest_point.x, furthest_point.y, label='furthest')
    plt.legend()
    plt.show()