#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import cv2
import numpy as np

global img
global point1, point2

lsPointsChoose = []
tpPointsChoose = []

pointsCount = 0
count = 0
pointsMax = 5

lsPointsChoose = []
tpPointsChoose = []

pointsCount = 0
count = 0
pointsMax = 5


def on_mouse(event, x, y, flags, param):
    global img, point1, point2, count, pointsMax
    global lsPointsChoose, tpPointsChoose  # 存入选择的点
    global pointsCount  # 对鼠标按下的点计数
    global init_img, ROI_bymouse_flag
    init_img = img.copy()  # 此行代码保证每次都重新再原图画  避免画多了

    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击

        pointsCount = pointsCount + 1
        # 为了保存绘制的区域，画的点稍晚清零
        if (pointsCount == pointsMax + 1):
            pointsCount = 0
            tpPointsChoose = []
        print('pointsCount:', pointsCount)
        point1 = (x, y)
        print(x, y)
        # 画出点击的点
        cv2.circle(init_img, point1, 10, (0, 255, 0), 5)

        # 将选取的点保存到list列表里
        lsPointsChoose.append([x, y])  # 用于转化为darry 提取多边形ROI
        tpPointsChoose.append((x, y))  # 用于画点

        # 将鼠标选的点用直线链接起来
        print(len(tpPointsChoose))
        for i in range(len(tpPointsChoose) - 1):
            cv2.line(init_img, tpPointsChoose[i], tpPointsChoose[i + 1], (0, 0, 255), 5)
        # 点击到pointMax时可以提取去绘图
        if (pointsCount == pointsMax):
            # 绘制感兴趣区域
            ROI_byMouse()
            ROI_bymouse_flag = 1
            lsPointsChoose = []

        cv2.imshow('src', init_img)

    # 右键按下清除轨迹
    if event == cv2.EVENT_RBUTTONDOWN:  # 右键点击
        print("right-mouse")
        pointsCount = 0
        tpPointsChoose = []
        lsPointsChoose = []
        print(len(tpPointsChoose))
        for i in range(len(tpPointsChoose) - 1):
            print('i', i)
            cv2.line(init_img, tpPointsChoose[i], tpPointsChoose[i + 1], (0, 0, 255), 5)
        cv2.imshow('src', init_img)


def ROI_byMouse():
    global src, ROI, ROI_flag, mask2
    mask = np.zeros(img.shape, np.uint8)
    pts = np.array([lsPointsChoose], np.int32)

    pts = pts.reshape((-1, 1, 2))  # -1代表剩下的维度自动计算

    # 画多边形
    mask = cv2.polylines(mask, [pts], True, (0, 255, 255))
    # 填充多边形
    mask2 = cv2.fillPoly(mask, [pts], (255, 255, 255))
    cv2.imshow('mask', mask2)
    ROI = cv2.bitwise_and(mask2, img)
    cv2.imshow('ROI', ROI)


def main():
    global img, init_img, ROI
    img = cv2.imread('test2.png')

    # 图像预处理，设置其大小
    height, width = img.shape[:2]
    size = (int(width * 0.3), int(height * 0.3))
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    ROI = img.copy()
    cv2.namedWindow('src')
    cv2.setMouseCallback('src', on_mouse)
    cv2.imshow('src', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
"""
import cv2

img=cv2.imread('test1.jpg')

#                         卷积和尺寸   是否归一化
#normalize=True  时等价于  均值滤波cv2.blur
#normalize=False 时 卷积结果>255时，置为255
blur=cv2.boxFilter(img,12, (3, 3), normalize=False)

cv2.imshow('boxFilter',blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""