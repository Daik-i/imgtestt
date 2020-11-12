import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv.imread('oo/biao.jpg')
height,width = img_rgb.shape[:2]
print(img_rgb.shape[:2])
cv.imshow('image',img_rgb)
cv.waitKey(0)
zero =  img_rgb*0
print(zero)
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
thre2 = thre1
# kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,5))
# open = cv.morphologyEx(thre1,cv.MORPH_OPEN,kernel,iterations=1)
# cv.imshow('image',open)
# cv.waitKey(0)
#
# circles = cv.HoughCircles(thre1, cv.HOUGH_GRADIENT, 2, 30, param1=200, param2=100, minRadius=20)
# print(circles)
# for circle in circles[0]:
#     center_x, center_y, radius = circle
#     cv.circle(thre1, (center_x, center_y), int(radius),(0, 0, 255), 2)
#
# cv.imshow("img", thre1)
# cv.waitKey(0)

roi_img1 = roi_img  = np.zeros(img_rgb.shape[0:2],dtype=np.uint8)
circles2 = cv.HoughCircles(img_gray, cv.HOUGH_GRADIENT_ALT, 2, 30, param1=300, param2=0.85, minRadius=20)
print(circles2)

for circle in circles2[0]:
    center_x, center_y, radius = circle
    cv.circle(img_rgb, (center_x, center_y), int(radius),(0, 0, 255), 2)
    cv.circle(img_rgb, (center_x, center_y), int(radius/2),(0, 255, 255), 2)
    cv.circle(roi_img, (center_x, center_y), int(radius),(255,255, 255), -1)
    cv.imshow("img2", img_rgb)
    cv.waitKey(0)

zero =  img_rgb*0
img_add_mask = cv.add(img_rgb,zero,mask=roi_img)
cv.imshow("img2", img_add_mask)
cv.waitKey(0)

canny_edge2 = cv.Canny(img_add_mask, threshold1=180, threshold2=230)
cv.imshow("img", canny_edge2)
cv.waitKey(0)

kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,5))
dst = cv.dilate(canny_edge2,kernel,iterations=1)
cv.imshow("img",dst)
cv.waitKey(0)

lines = cv.HoughLines(dst,1,np.pi/180,int(radius*0.65))
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv.line(img_rgb,(x1,y1),(x2,y2),(0,0,255),2)
    print(line)
cv.imshow("img11", img_rgb)
cv.waitKey(0)
# cv.imwrite('houghlines3.jpg',thre1)


# sobel_edge = cv.Sobel(img_rgb,ddepth=cv.CV_32F,dx=1,dy=1,ksize=5)
# cv.imshow("img", sobel_edge)
# cv.waitKey(0)

# circles = cv.HoughCircles(canny_edge2, cv.HOUGH_GRADIENT_ALT, 2, 30, param1=300, param2=0.85, minRadius=20)
# print(circles)
#
# for circle in circles[0]:
#     center_x, center_y, radius = circle
#     cv.circle(thre2, (center_x, center_y), int(radius),(0, 0, 255), 2)
#     cv.circle(roi_img1, (center_x, center_y), int(radius),(255,255, 255), -1)
# cv.imshow("img2", thre2)
# cv.waitKey(0)
#
#
# img_add_mask1 = cv.add(img_rgb,zero,mask=roi_img1)
# cv.imshow("img2", img_add_mask1)
# cv.waitKey(0)