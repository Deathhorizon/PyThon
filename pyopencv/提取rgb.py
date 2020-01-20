#!/usr/bin/env python 
# -*- coding:utf-8 -*-
#本程序用于将一张彩色图片分解成BGR的分量显示，灰度图显示，HSV分量显示
import cv2  #导入opencv模块
import numpy as np

print("Hellow word!")     #打印“Hello word！”，验证模块导入成功

img = cv2.imread("test2.png")  #导入图片，图片放在程序所在目录
cv2.namedWindow("imagshow", 2)   #创建一个窗口
cv2.imshow('imagshow', img)    #显示原始图片

"""
#使用直接访问的方法
B = img[:, :, 0]
G = img[:, :, 1]
R = img[:, :, 2]
"""
#使用split函数分解BGR
(B, G, R) = cv2.split(img) #分离图像的RBG分量

#以灰度图的形式显示每个颜色的分量
"""
cv2.namedWindow("B",2)   #创建一个窗口
cv2.imshow('B', B)       #显示B分量
cv2.namedWindow("G",2)   #创建一个窗口
cv2.imshow('G', G)       #显示G分量
cv2.namedWindow("R",2)   #创建一个窗口
cv2.imshow('R', R)       #显示R分量
"""

# 生成一个值为0的单通道数组
zeros = np.zeros(img.shape[:2], dtype = "uint8")
# 分别扩展B、G、R成为三通道。另外两个通道用上面的值为0的数组填充
cv2.namedWindow("Blue",2)   #创建一个窗口
cv2.imshow("Blue", cv2.merge([B, zeros, zeros]))
cv2.namedWindow("Green",2)   #创建一个窗口
cv2.imshow("Green", cv2.merge([zeros, G, zeros]))
cv2.namedWindow("Red",2)   #创建一个窗口
cv2.imshow("Red", cv2.merge([zeros, zeros, R]))

#使用cvtColor转换为灰度图
out_img_GRAY=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#将图片转换为灰度图
cv2.namedWindow("GRAY_imag",2)   #创建一个窗口
cv2.imshow('GRAY_imag', out_img_GRAY) #显示灰度图

#使用cvtColor转换为HSV图
out_img_HSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)#将图片转换为灰度图
hsvChannels=cv2.split(out_img_HSV)  #将HSV格式的图片分解为3个通道

cv2.namedWindow("Hue",2)   #创建一个窗口
cv2.imshow('Hue',hsvChannels[0]) #显示Hue分量
cv2.namedWindow("Saturation",2)   #创建一个窗口
cv2.imshow('Saturation',hsvChannels[1]) #显示Saturation分量
cv2.namedWindow("Value",2)   #创建一个窗口
cv2.imshow('Value',hsvChannels[2]) #显示Value分量

cv2.waitKey(0)  #等待用户操作
