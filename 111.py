#coding:utf-8

import cv2 as cv
import matplotlib.pyplot as plt

# img = cv.imread(r"oo/23.jpg",0)
#
# ret,thre1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
# adaptive_thre1 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,7,2)
# adaptive_thre2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,7,2)
#
# titles = ["img","thre1","adaptive_thre1","adaptive_thre2"]
# imgs = [img,thre1,adaptive_thre1,adaptive_thre2 ]
# cv.imwrite(r"oo/num8.jpg",thre1)
# for i in range(4):
#     plt.subplot(2,2,i+1), plt.imshow(imgs[i],"gray")
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()



import numpy as np
import cv2 as cv
drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            else:
                cv.circle(img,(x,y),5,(0,0,255),-1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
        else:
            cv.circle(img,(x,y),5,(0,0,255),-1)
img = np.zeros((512,512,3), np.uint8)
cv.namedWindow('image')
cv.setMouseCallback('image',draw_circle)
while(1):
    cv.imshow('image',img)
    k = cv.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break
cv.destroyAllWindows()