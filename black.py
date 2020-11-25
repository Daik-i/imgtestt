import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

threshold = 0.50
template = [[] for i in range(20)]
for i in range(20):
    template[i] = cv.imread('oo/num'+str(i)+'.jpg', 0)
    # cv.imshow('image', template[i])
    # cv.waitKey(0)
# print(template)
w, h = template[0].shape[::-1]
print(w ,h)

def match(img,x,y):
    global threshold,template,img_rgb,cutx, cuty
    reslut = []
    for i in range(20):
        resl = cv.matchTemplate(img, template[i], cv.TM_CCOEFF_NORMED)
        print(resl)
        reslut.append(resl[0][0])
    print(reslut)
    if (max(reslut, key = abs) >= threshold):
        num = int(reslut.index(max(reslut, key = abs))/2)
        cv.rectangle(img_rgb, (cutx[y], cuty[x]), (cutx[y + 1], cuty[x + 1]), (0, 0, 255), 2)
    else:
        num = -1
    return num

img_rgb = cv.imread('oo/25.jpg')
cv.imshow('image',img_rgb)
cv.waitKey(0)

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

# 灰度值在水平方向上的投影
county = [0 for i in range(height)]
for i in range(height):
    for j in thre1[i]:
        if j != 0:
            county[i] +=1
# print(county)
# 灰度值在垂直方向上的投影
countx = [0 for i in range(width)]
for i in range(width):
    for j in thre1.T[i]:
        if j != 0:
            countx[i] +=1
# print(countx)

# 水平方向和垂直方向上的灰度投影统计
# print(countx.append([0]))
plt.subplot(1,2,1), plt.plot(range(height),county)
plt.subplot(1,2,2), plt.plot(range(width),countx)
# plt.xticks([]),plt.yticks([])
plt.show()

# 找出数码的四个角
cuty = []
for i in range(height-1):
    if (county[i] == 0 and county[i+1] != 0) :
        cuty.append(i+1)
    elif (county[i] != 0 and county[i+1] == 0):
        cuty.append(i)
print(cuty)

# 数码规格不一样，不能由高度计算宽度
cutx = []
for i in range(width-1):
    if (countx[i] == 0 and countx[i + 1] != 0):
        cutx.append(i + 1)
    elif (countx[i] != 0 and countx[i + 1] == 0):
        cutx.append(i)
print(cutx)



numm = np.zeros((int(len(cuty) / 2), int(len(cutx) / 2)), dtype=np.int)

for i in range(0,int(len(cuty)),2):
    for j in range(0,int(len(cutx)-1),2):
        roi = thre1[cuty[i]:cuty[i+1]+1,cutx[j]:cutx[j+1]+1]
        res = cv.resize(roi,(w, h),0,0, interpolation = cv.INTER_CUBIC)
        # print(roi)
        cv.imshow('image', roi)
        cv.waitKey(0)
        # print(res)
        numm[int(i/2)][int(j/2)] += match(res,i,j)
        # cv.imwrite('res83.png',img_rgb)
        # cv.imshow('image', roi)
        # cv.waitKey(0)
print(numm)
cv.imshow('image', img_rgb)
cv.waitKey(0)