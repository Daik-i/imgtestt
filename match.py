import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img_rgb = cv.imread('oo/22.jpg')
cv.imshow('image',img_rgb)
cv.waitKey(0)
# img_rgb = cv.resize(img_rgb,None,fx=2, fy=2, interpolation = cv.INTER_CUBIC)
# cv.imshow('image',img_rgb)
# cv.waitKey(0)

wid, hit,lu = img_rgb.shape
print(img_rgb)
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
print(img_gray)
# cv.imshow('image',img_gray)
# cv.waitKey(0)
# img_gray = cv.resize(img_gray,None,fx=2, fy=2, interpolation = cv.INTER_CUBIC)
# cv.imshow('image',img_gray)
# cv.waitKey(0)
adaptive_thre1 = cv.adaptiveThreshold(img_gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,7,2)
print(adaptive_thre1)
print(adaptive_thre1.T)
countx = [0 for i in range(wid)]
print(countx)
for i in range(wid):
    for j in adaptive_thre1[i]:
        if j != 0:
            countx[i] +=1
print(countx)
widy, hity = adaptive_thre1.T.shape
county = [0 for i in range(widy)]
for i in range(widy):
    for j in adaptive_thre1.T[i]:
        if j != 0:
            county[i] +=1
print(county)


plt.subplot(1,2,1), plt.plot(range(wid),countx)
plt.subplot(1,2,2), plt.plot(range(widy),county)
# plt.xticks([]),plt.yticks([])
plt.show()

template = cv.imread('oo/num8.jpg',0)
# template = cv.resize(template,None,fx=2, fy=2, interpolation = cv.INTER_CUBIC)
w, h = template.shape[::-1]
print(w, h)
res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
print(loc)
for pt in zip(*loc[::-1]):
    print(pt)
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
# cv.imwrite('res83.png',img_rgb)
cv.imshow('image',img_rgb)
cv.waitKey(0)