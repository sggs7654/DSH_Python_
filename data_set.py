import numpy as np
import math
import matplotlib.pyplot as plt


class Point:
    x = 0
    y = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def dist(self, point):
        delta_x = self.x - point.x
        delta_y = self.y - point.y
        return math.sqrt(math.pow(delta_x, 2)+math.pow(delta_y, 2))


class PointSet:
    seed = 1
    point_num = 80
    point_set = []
    center_num = 5
    center_set = []
    radius = 20
    min_coordinate = 0
    max_coordinate = 150

    def __init__(self):
        self.build_center()
        self.build_points()

    def reset_seed(self):
        self.seed = self.seed + 11
        np.random.seed(self.seed)

    def build_center(self):
        for i in range(self.center_num):
            while True:  # 产生目标区域内的坐标点
                self.reset_seed()
                x = np.random.randint(low=self.min_coordinate + self.radius,
                                      high=self.max_coordinate - self.radius)
                self.reset_seed()
                y = np.random.randint(low=self.min_coordinate + self.radius,
                                      high=self.max_coordinate - self.radius)
                center_point = Point(x, y)
                if len(self.center_set) == 0:  # 如果这是第一个中心点,则不需要碰撞检测
                    break
                # 对于后续的中心点,则需要检测它们是否与其他已有的中心点发生碰撞
                collision = False
                for j in range(len(self.center_set)):
                    if center_point.dist(self.center_set[j]) < 2 * self.radius:  # 碰撞定义:半径重叠
                        collision = True
                if not collision:
                    break  # 如果新点与任意一个中心点发生碰撞, 则需要重新产生新点
            self.center_set.append(center_point)

    def build_points(self):
        generate_count = 0
        while True:
            if generate_count >= self.point_num:
                return
            # 随机选择一个中心点, 作为center_point
            self.reset_seed()
            center_index = np.random.randint(low=0, high=self.center_num)
            center_point = self.center_set[center_index]
            cx = center_point.x
            cy = center_point.y
            self.reset_seed()
            x = np.random.normal(loc=cx, scale=self.radius)
            # x = np.random.uniform(low=cx-self.radius, high=cx+self.radius)
            self.reset_seed()
            y = np.random.normal(loc=cy, scale=self.radius)
            # y = np.random.uniform(low=cy-self.radius, high=cy+self.radius)
            new_point = Point(x, y)
            self.point_set.append(new_point)
            generate_count = generate_count + 1

    def get_matrix(self):
        x = []
        y = []
        for i in range(self.point_num):
            x.append(self.point_set[i].x)
            y.append(self.point_set[i].y)
        # plt.scatter(x, y)
        # plt.show()
        x = np.array([x]).transpose()
        y = np.array([y]).transpose()
        return np.concatenate((x, y), axis=1)

    def show(self):
        x = []
        y = []
        for i in self.center_set:
            x.append(i.x)
            y.append(i.y)
        plt.scatter(x, y, label='center')
        x.clear()
        y.clear()
        for i in self.point_set:
            x.append(i.x)
            y.append(i.y)
        plt.scatter(x, y, label='scatter')
        plt.legend()
        plt.show()
