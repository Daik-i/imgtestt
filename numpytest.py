import numpy as np
import cv2 as cv

img_rgb = cv.imread('oo/22.jpg')
# cv.imshow('image',img_rgb)
# cv.waitKey(0)
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
juzhen = np.array(img_gray,dtype=int)
print(juzhen)
print (juzhen[ ...,1])

a = np.arange(24)
print (a.ndim)             # a 现只有一个维度
# 现在调整其大小
b = a.reshape(2,4,3)  # b 现在拥有三个维度
print (b)
print (a.flags)
print (b.flags)

list = range(5)
it = iter(list)

a = np.linspace(10, 20,  5, endpoint = True)
print(a)