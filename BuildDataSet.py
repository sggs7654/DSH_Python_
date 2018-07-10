import numpy as np
import math
import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans


class Point:
    x = 0
    y = 0

    def __init__(self,x,y):
        self.x = x
        self.y = y

    def dist(self,point):
        deltaX = self.x - point.x
        deltaY = self.y - point.y
        return math.sqrt(math.pow(deltaX,2)+math.pow(deltaY,2))


class PointSet:
    seed = 2
    point_num = 40
    point_set = []
    center_num = 4
    center_set = []
    radius = 10
    min_coordinate = 0
    max_coordinate = 150

    def reset_seed(self):
        self.seed = self.seed + 11
        np.random.seed(self.seed)

    def build_center(self):
        for i in range(0, self.center_num):
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
                for i in range(0, len(self.center_set)):
                    if center_point.dist(self.center_set[i]) < 2 * self.radius:  # 碰撞定义:半径重叠
                        collision = True
                if not collision: break  # 如果新点与任意一个中心点发生碰撞, 则需要重新产生新点
            self.center_set.append(center_point)

    def build_points(self):
        cluster_size = int(self.point_num / self.center_num)
        generate_count = 0
        for center_point in self.center_set:
            for i in range(0,cluster_size):
                if generate_count >= self.point_num:
                    return
                cx = center_point.x
                cy = center_point.y
                self.reset_seed()
                x = np.random.uniform(low=cx-self.radius, high=cx+self.radius)
                self.reset_seed()
                y = np.random.uniform(low=cy-self.radius, high=cy+self.radius)
                new_point = Point(x,y)
                self.point_set.append(new_point)
                generate_count = generate_count + 1

    def points_show(self):
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


pointSet = PointSet()
pointSet.build_center()
pointSet.build_points()
pointSet.points_show()
# print(pointSet.x)
# print(pointSet.y)
# plt.scatter(pointSet.x, pointSet.y)
# plt.show()
