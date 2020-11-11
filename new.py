import cv2 as cv
import numpy as np

template = cv.imread('oo/numw1.jpg', 0)
res = cv.resize(template,(85, 150),0,0, interpolation = cv.INTER_CUBIC)
ret,thre1 = cv.threshold(res,127,255,cv.THRESH_BINARY)
cv.imshow('image',thre1)
cv.waitKey(0)
cv.imwrite('oo/num01.jpg',res)