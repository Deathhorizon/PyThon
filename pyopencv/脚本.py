#!/usr/bin/env python 
# -*- coding:utf-8 -*-
#导入cv模块
import cv2
# 读取图像，支持 bmp、jpg、png、tiff 等常用格式

img = cv2.imread("20181213134139.png")
# 创建窗口并显示图像
cv2.namedWindow("Image", 10)
cv2.imshow("Image", img)
cv2.waitKey(10000)
# 释放窗口
cv2.destroyAllWindows()
