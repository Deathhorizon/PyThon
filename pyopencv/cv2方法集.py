#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import cv2
img1= cv2.imread('book.jpg')
img2=cv2.imread('book.jpg')
img = cv2.absdiff(img1,img2 )
#cv2.cvtColor(img1,)
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('img', img)

cv2.waitKey(0)
cv2.destroyAllWindows()