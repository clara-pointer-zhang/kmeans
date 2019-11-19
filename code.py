#!/usr/bin/python
import sys
sys.path.append(r'c:\users\administrator\appdata\local\programs\python\python37\lib\site-packages')
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from sklearn.cluster import KMeans


#计算距离
def distance(p, xlist = [], ylist = []):
    sum = 0
    for i in range(2):
        sum += (abs(xlist[i] -ylist[i])) ** p
    sum = sum ** (1 / p)
    return sum

#找到聚类中心之间的最大距离
def maxdistance(p, center = [], data = []):
    m = len(center)
    maxdist = 0
    for i in range(0, m, 1):
        for j in range(i+1, m, 1):
            tmp = distance(p, data[center[i]], data[center[j]])
            if tmp > maxdist:
                maxdist = tmp
    return maxdist

#在非中心的点中挑选与对应聚类中心距离最大的点，并把距离和聚类中心之间的最大距离进行比较
def searchnewcenter(p, data = [], center = [], dist = [], cluster = []):
    for i in range(len(data)):
        if dist[i] != -1:
            tmp = distance(p, data[center[-1]], data[i])
            if tmp < dist[i]:
                dist[i] = tmp
                cluster[i] = center[-1]
    maxdist = max(dist)
    k = int(dist.index(maxdist))
    if maxdist >= thre * maxdistance(p, center, data):#如果找到聚类中心继续找
        center.append(k)
        dist[k] = -1
        cluster[k] = k
        searchnewcenter(p, data, center, dist, cluster)
        return 1
    else:#结束寻找
        return 0

#根据不同聚类分配颜色和形状画图
def draw_scatter(n, s, center=[], cluster=[], data=[]):
    colorlist = ['#000000', '#0000FF', '#A52A2A', '#D2691E', '#FF7F50', '#B8860B', '#FF8C00', '#E9967A',
                '#808080', '#FFFF00', '#DDA0DD', '#2E8B57', '#FF6347', '#DC143C', '#00CED1']
    markerlist = [".", "o", "v", "^", "1", "8", "s", "*", "+", "x", "d", "<", ">", "2", "p"]
    #将数据按照聚类分开
    x = []
    y = []
    for i in range((len(center))):
        x.append([])
        y.append([])
    for i in range(n):
        index = center.index(cluster[i])
        x[index].append(data[i][0])
        y[index].append(data[i][1])
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title('aggregation')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    for i in range(len(center)):
        ax1.scatter(x[i], y[i], s=s, c=colorlist[i], marker=markerlist[i])
    plt.xlim(xmax=40, xmin=0)
    plt.show()

#算法4,5用来画图
def pict(k, x = [], y = [], label_pred = []):
    colorlist = ['#000000', '#0000FF', '#A52A2A', '#D2691E', '#FF7F50', '#B8860B', '#FF8C00', '#E9967A',
                 '#808080', '#FFFF00', '#DDA0DD', '#2E8B57', '#FF6347', '#DC143C', '#00CED1']
    markerlist = [".", "o", "v", "^", "1", "8", "s", "*", "+", "x", "d", "<", ">", "2", "p"]
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title('aggregation')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    for i in range(k):
        ax1.scatter(x[i], y[i], s=20, c=colorlist[i], marker = markerlist[i])
    plt.xlim(xmax=40, xmin=0)
    plt.show()

#算法0,5利用sklearn中的包kmeans
def clustern(k, data=[]):
    estimator = KMeans(n_clusters=k)  # 构造聚类器
    estimator.fit(data)  # 聚类
    label_pred = estimator.labels_  # 获取聚类标签
    #初始化x，y
    x = []
    y = []
    for i in range(k):
        x.append([])
        y.append([])
    for i in range(len(data)):  #根据label分开数据
        x[label_pred[i]].append(data[i][0])
        y[label_pred[i]].append(data[i][1])
    print(label_pred)
    pict(k, x, y, label_pred)


#为算法4改进的maxdistance函数，因为算法4中的聚类中心并不是data中的数据
def maxdistancefor4(p, position = []):
    m = len(position)
    maxdist = 0
    for i in range(0, m, 1):
        for j in range(i+1, m, 1):
            tmp = distance(p, position[i], position[j])
            if tmp > maxdist: maxdist = tmp
    return maxdist

#为算法4改进的searchnewcenter函数
def searchnewcenterfor4(p, thre, position = [], dist = []):
    maxdist = max(dist)
    if maxdist >= thre * maxdistancefor4(p, position):
        return 1
    else:
        return 0

#算法4中的重复kmeans直到不再增加新聚类中心的过程
def cluster(k, p, thre, data = [], dist = []):
    estimator = KMeans(n_clusters=k)  # 构造聚类器
    estimator.fit(data)  # 聚类
    label_pred = estimator.labels_  # 获取聚类标签
    x = []
    y = []
    for i in range(k): #将数据按照聚类分类便于画图
        x.append([])
        y.append([])
    for i in range(len(data)):
        x[label_pred[i]].append(data[i][0])
        y[label_pred[i]].append(data[i][1])
    position = []    #更新聚类中心
    for i in range(k):
        sum1 = 0.0
        sum2 = 0.0
        for j in range(len(x[i])):
            sum1 += x[i][j]
            sum2 += y[i][j]
        position.append([sum1/len(x[i]), sum2/len(x[i])])
    for i in range(len(label_pred)): #更新dist列表
        ylist = []
        ylist.append(data[i][0])
        ylist.append(data[i][1])
        l = label_pred[i]
        dist[i] = distance(p, position[l], ylist)
    if searchnewcenterfor4(p, thre, position, dist) == 1:
        k += 1
        cluster(k, p, thre, data, dist)
    else:
        print(k)
        pict(k, x, y, label_pred)



if __name__ == '__main__':
    print("选择算法")
    mod = int(input())
    print("输入阈值和范数")   #阈值用来约束聚类中心的生成，范数改变距离计算方式
    thre, p = map(float, input().split())
    center = []
    data = []
    for line in open("E:\Aggregation.txt", "r"):   #将数据读入data列表中
        xlist = line.split()
        xlist = [float(xlist[i]) for i in range(len(xlist))]
        data.append(xlist)
    m = len(data)
    #数据自带label画图
    if mod == -1:
        colorlist = ['#000000', '#0000FF', '#A52A2A', '#D2691E', '#FF7F50', '#B8860B', '#FF8C00', '#E9967A',
                     '#808080', '#FFFF00', '#DDA0DD', '#2E8B57', '#FF6347', '#DC143C', '#00CED1']
        markerlist = [".", "o", "v", "^", "1", "8", "s", "*", "+", "x", "d", "<", ">", "2", "p"]
        # 将数据按照聚类分开
        x = []
        y = []
        label = []
        for i in range(m):
            label.append(int(data[i][2]))
        n = max(label)
        for i in range(n):
            x.append([])
            y.append([])
        for i in range(m):
            x[label[i]-1].append(data[i][0])
            y[label[i]-1].append(data[i][1])
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('aggregation')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        for i in range(n):
            ax1.scatter(x[i], y[i], s=20, c=colorlist[i], marker=markerlist[i])
        plt.xlim(xmax=45, xmin=0)
        plt.show()
    #原始kmeans
    elif mod == 0:
        print("输入聚类个数")
        k = int(input())
        clustern(k, data)
    #第一种算法改进后的kmeans
    elif mod == 1:
        # 随机选取一个聚类中心
        k = random.randint(0, m - 1)
        # 初始化聚类中心，距离和聚类标签列表
        center.append(k)
        dist = [10000 for i in range(m)]
        dist[k] = -1
        cluster = [-1 for i in range(m)]
        cluster[k] = k
        # 寻找聚类中心
        searchnewcenter(p, data, center, dist, cluster)
        # 输出聚类个数
        print(len(center))
        print(cluster)
        # 画图
        draw_scatter(len(cluster), 20, center, cluster, data)

    #第二种算法一开始选择均值作为聚类中心
    elif mod == 2:
        sum1 = 0.0
        sum2 = 0.0
        for i in range(m):
            sum1 += data[i][0]
            sum2 += data[i][1]
        data.append([sum1 / m, sum2 / m])
        center.append(m)
        dist = [10000 for i in range(m + 1)]
        dist[m] = -1
        cluster = [-1 for i in range(m + 1)]
        cluster[m] = m
        searchnewcenter(p, data, center, dist, cluster)
        print(len(center))
        draw_scatter(len(cluster) - 1, 20, center, cluster, data)

    #第三种算法使用一开始距离均值中心距离最大的点作为聚类中心
    elif mod == 3:
        sum1 = 0.0
        sum2 = 0.0
        tmp = []
        for i in range(m):
            sum1 += data[i][0]
            sum2 += data[i][1]
        tmp.append(sum1 / m)
        tmp.append(sum2 / m)
        dist = [10000 for i in range(m)]
        cluster = [-1 for i in range(m)]
        for i in range(len(data)):
            dist[i] = distance(p, tmp, data[i])
        maxdist = max(dist)
        k = int(dist.index(maxdist))
        dist[k] = -1
        cluster = [k for i in range(m)]
        center.append(k)
        searchnewcenter(p, data, center, dist, cluster)
        print(len(center))
        draw_scatter(len(cluster), 20, center, cluster,data)
    #第四种算法先kmeans再通过阈值计算是否需要增加聚类
    elif mod == 4:
        print("输入聚类个数")
        k = int(input())
        dist = [10000 for i in range(m)]
        cluster(k, p, thre, data, dist)
    #第五种算法在得出聚类中心个数后再进行kmeans
    elif mod == 5:
        k = random.randint(0, m - 1)
        center.append(k)
        dist = [10000 for i in range(m)]
        dist[k] = -1
        cluster = [-1 for i in range(m)]
        cluster[k] = k
        searchnewcenter(p, data, center, dist, cluster)
        print(len(center))
        clustern(len(center), data)
    else:
        print("还没想出更多算法")
