#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread('test2.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
x, y, w, h = cv2.boundingRect(contours[1])
#for i in range(0,int(len((contours)))):
    #print(len(contours[i]))
#cv2.polylines(img,contours[2],True,(0,255,255))
cv2.rectangle(img, (x,y), (x+w,y+h), (153,153,0),1)
newimage = img[y + 0:y + h - 0, x + 0:x + w-0]  # 先用y确定高，再用x确定宽
cv2.imwrite( "12.jpg",newimage)
cv2.imwrite( "13.jpg",img)
cv2.imshow('img', img)
cv2.waitKey(0)
"""
#                         卷积和尺寸   是否归一化
#normalize=True  时等价于  均值滤波cv2.blur
#normalize=False 时 卷积结果>255时，置为255
blur=cv2.boxFilter(img,-100, (3, 3), normalize=False)

cv2.imshow('boxFilter',blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread('ball.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.subplot(121), plt.imshow(gray, 'gray')
plt.xticks([]), plt.yticks([])

circles1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,
                            600, param1=100, param2=30, minRadius=80, maxRadius=97)
circles = circles1[0, :, :]
circles = np.uint16(np.around(circles))
for i in circles[:]:
    cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 5)
    cv2.circle(img, (i[0], i[1]), 2, (255, 0, 255), 10)
    cv2.rectangle(img, (i[0] - i[2], i[1] + i[2]), (i[0] + i[2], i[1] - i[2]), (255, 255, 0), 5)

print("圆心坐标", i[0], i[1])
plt.subplot(122), plt.imshow(img)
plt.xticks([]), plt.yticks([])

import cv2
import numpy

im = cv2.imread('test.jpg')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# 绘制独立轮廓，如第四个轮廓：
img = cv2.drawContour(im, contours, -1, (0,255,0), 3)
# 但是大多数时候，下面的方法更有用：
img = cv2.drawContours(img, contours, 3, (0,255,0), 3)

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('ball.png', 0)
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks(), plt.yticks()  # to hide tick values on X and Y axis
plt.plot([500, 500, 400], [100, 200, 300], 'b', linewidth=2)
plt.show()"""
"""
import matplotlib.pyplot as plt
import numpy as np

'''read file 
fin=open("para.txt")
a=[]
for i in fin:
  a.append(float(i.strip()))
a=np.array(a)
a=a.reshape(9,3)
'''
a = np.random.random((9, 3)) * 2  # 随机生成y

y1 = a[0:, 0]
y2 = a[0:, 1]
y3 = a[0:, 2]

x = np.arange(1, 10)

ax = plt.subplot(111)
width = 10
hight = 3
ax.arrow(0, 0, 0, hight, width=0.01, head_width=0.1, head_length=0.3, length_includes_head=True, fc='k', ec='k')
ax.arrow(0, 0, width, 0, width=0.01, head_width=0.1, head_length=0.3, length_includes_head=True, fc='k', ec='k')

ax.axes.set_xlim(-0.5, width + 0.2)
ax.axes.set_ylim(-0.5, hight + 0.2)

plotdict = {'dx': x, 'dy': y1}
ax.plot('dx', 'dy', 'bD-', data=plotdict)

ax.plot(x, y2, 'r^-')
ax.plot(x, y3, color='#900302', marker='*', linestyle='-')

plt.show()

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 2 * np.pi, 0.02)
y = np.sin(x)
y1 = np.sin(2 * x)
y2 = np.sin(3 * x)
ym1 = np.ma.masked_where(y1 > 0.5, y1)
ym2 = np.ma.masked_where(y2 < -0.5, y2)

lines = plt.plot(x, y, x, ym1, x, ym2, 'o')
# 设置线的属性
plt.setp(lines[0], linewidth=1)
plt.setp(lines[1], linewidth=2)
plt.setp(lines[2], linestyle='-', marker='^', markersize=4)
# 线的标签
plt.legend(('No mask', 'Masked if > 0.5', 'Masked if < -0.5'), loc='upper right')
plt.title('Masked line demo')
plt.show()
help(plt.plot)"""
"""
cap = cv2.VideoCapture(0)
while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):break
cap.release()
cv2.destroyAllWindows()
"""
'''录制视频并保存
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out.write(frame)
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
out.release()
cv2.destroyAllWindows()'''

"""img = cv2.imread('1529393771416244b60d277.jpg')

retval, threshold = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)
cv2.imshow('original',img)
cv2.imshow('threshold',threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()

grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
retval, threshold = cv2.threshold(grayscaled, 10, 255, cv2.THRESH_BINARY)
cv2.imshow('original',img)
cv2.imshow('threshold',threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()
grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
th = cv2.adaptiveThreshold(grayscaled, 127, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
cv2.imshow('original',img)
cv2.imshow('Adaptive threshold',th)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
#help(cv2.threshold)