import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv.imread('oo/biao.jpg')
height,width = img_rgb.shape[:2]
print(img_rgb.shape[:2])

# 取红色通道
# img_r = img_rgb[:,:,2]
# print(img_r)
# cv.imshow('image',img_r)
# cv.waitKey(0)

# 高斯平滑去噪
img_gauss = cv.GaussianBlur(img_rgb,(3,3),1)
cv.imshow('image',img_gauss)
cv.waitKey(0)

# 转为灰度图
img_gray = cv.cvtColor(img_gauss,cv.COLOR_RGB2GRAY)
cv.imshow('image',cv.cvtColor(img_rgb,cv.COLOR_RGB2GRAY))
cv.waitKey(0)

# #阀值化
ret,thre1 = cv.threshold(img_gray,185,255,cv.THRESH_BINARY)
# thre1 = cv.adaptiveThreshold(img_gauss,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,7,2)
cv.imshow('image',thre1)
cv.waitKey(0)