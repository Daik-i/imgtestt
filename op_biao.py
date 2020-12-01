import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 给出线性方程组并求解
def get_cross_point(x,y,z):
    A = np.array([x, y], dtype = float)
    B = np.array(z)
    res_m = np.linalg.solve(A.T, B)
    print(res_m)
    return res_m

# 通过距离和角度画出直线，返回其中两点和距离角度
def get_line_point(line,img):
    rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return x1,y1,x2,y2,rho, theta

# 得到两点的角度
def get_point(event,x,y,flags,param):
    global point_dst,start_theta,end_theta
    if event == cv.EVENT_LBUTTONDOWN :
        cv.circle(img_rgb,(x,y),3,(255,0,0),-1)
        print(x, y)
        point_dst.append([x,y])
        print(point_dst)
        if len(point_dst) == 2:
            point1 = point_dst[-2]
            point2 = point_dst[-1]
            # cv.line(img_rgb, tuple(point1),tuple(point2), (0, 0, 255), 2)
            start_theta = np.degrees(np.arctan((center_y - point1[1]) / (point1[0] - center_x))) -180
            print(start_theta)
            end_theta = np.degrees(np.arctan((center_y - point2[1]) / (point2[0] - center_x))) +180
            print(end_theta)

point_dst = []
start_theta = 0
end_theta = 0
img_rgb = cv.imread('oo/moban.jpg')
cv.namedWindow('image')
cv.setMouseCallback('image',get_point)
height,width = img_rgb.shape[:2]
print(img_rgb.shape[:2])
cv.imshow('image',img_rgb)
cv.waitKey(0)
zero =  img_rgb*0

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
cv.imshow('image',img_gray)
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


# 找出的圆里画出其边界，圆心，以及构造出以此圆为目标的掩码
for circle in circles2[0]:
    center_x, center_y, radius = circle
    cv.circle(img_rgb, (center_x, center_y), int(radius),(0, 0, 255), 2)
    cv.circle(img_rgb, (center_x, center_y),3,(0, 0, 255), -1)
    cv.circle(roi_img, (center_x, center_y), int(radius),(255,255, 255), -1)
    # cv.imshow("roi_img", roi_img)
    cv.imshow("image", img_rgb)
    cv.waitKey(0)

# cv.add(img1,img2,mask)将两图进行相加运算并与掩码mask相与
zero =  img_rgb*0
img_add_mask = cv.add(img_rgb,zero,mask=roi_img)
cv.imshow("image", img_add_mask)
cv.waitKey(0)

canny_edge2 = cv.Canny(img_add_mask, threshold1=180, threshold2=230)
cv.imshow("image", canny_edge2)
cv.waitKey(0)

# kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,5))
# dst = cv.dilate(canny_edge2,kernel,iterations=1)
# cv.imshow("img",dst)
# cv.waitKey(0)

'''
cv.HoughLines(canny_edge2,1,np.pi/180,int(radius*0.22))
    image: 单通道的灰度图或二值图
    rho: 距离步长，单位为像素(上述投票器中纵轴)
    theta: 角度步长，单位为弧度 (上述投票器中横轴)
    threshold: 投票器中计数阈值，当投票器中某个点的计数超过阈值，则认为该点对应图像中的一条直线，也可以理解为图像空间空中一条直线上的像素点个数阈值(如设为5，则表示这条直线至少包括5个像素点)
    srn:默认为0， 用于在多尺度霍夫变换中作为参数rho的除数，rho=rho/srn
    stn：默认值为0，用于在多尺度霍夫变换中作为参数theta的除数，theta=theta/stn
        (如果srn和stn同时为0，就表示HoughLines函数执行标准霍夫变换，否则就是执行多尺度霍夫变换)
    min_theta: 默认为0，表示直线与坐标轴x轴的最小夹角为0
    max_theta：默认为CV_PI，表示直线与坐标轴x轴的最大夹角为180度
'''
line_K = []
line_b = []
print(radius)
lines = cv.HoughLines(canny_edge2,1,np.pi/180,int(radius*0.22))  #1>=0.72   3 <=0.65
print(lines)
if len(lines) == 0:
    print('未检测到指针')
elif len(lines) == 1:
    x1, y1, x2, y2, rho, theta = get_line_point(lines[0][0],img_rgb)
    line_theta = 90 - np.degrees(np.arctan((center_y - y1) / (x1 - center_x)))  #  度数需要再考虑一下
    print(line_theta)
    print('请选择初始位置和终止位置')
elif len(lines) == 2:
    for line in lines:
        x1, y1, x2, y2,rho, theta = get_line_point(line[0],img_rgb)
        A = [x1,x2]
        B = [y1,y2]
        kb = get_cross_point(A,[1,1],B)
        line_K.append(kb[0])
        line_b.append(kb[1])
    xy = get_cross_point([line_K[0]*-1,line_K[1]*-1],[1,1],line_b)
    cv.circle(img_rgb, (int(xy[0]), int(xy[1])), 3, (255, 0, 0), -1)
    if center_x <= xy[0]:
        line_theta = 90 - np.degrees(np.arctan((center_y - xy[1]) / (xy[0] - center_x)))
    else:
        line_theta = -90 - np.degrees(np.arctan((center_y - xy[1]) / (xy[0] - center_x)))
    print(line_theta)
    print('请选择初始位置和终止位置')
else:
    for line in lines:
        x1, y1, x2, y2,rho, theta = get_line_point(line[0],img_rgb)
    print('指针检测不准确，请重新检测')




cv.imshow("image", img_rgb)
cv.waitKey(0)
print((line_theta-start_theta),(end_theta-start_theta))
res_dst = (line_theta-start_theta)/(end_theta-start_theta)*1.6
print(res_dst)
cv.imshow("image", img_rgb)
cv.waitKey(0)

# while True:
#     if len(point_dst) == 2:





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