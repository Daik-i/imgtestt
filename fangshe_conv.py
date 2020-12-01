import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# mouse callback function


# 将一个图片进行仿射变换
def draw_circle(event,x,y,flags,param):
    global dst
    if event == cv.EVENT_LBUTTONDBLCLK:
        cv.circle(img,(x,y),3,(255,0,0),-1)
        print(type(x), y)
        place.append([x,y])
        print(place)
        print(len(place))
        if len(place) == 4:
            list1 = [place[0][0], place[1][0], place[2][0], place[3][0]]
            list2 = [place[0][1], place[1][1], place[2][1], place[3][1]]
            print(list1)
            print(list2)
            place_left = place[list1.index(min(list1, key = abs))]
            place_right = place[list1.index(max(list1, key = abs))]
            place_up = place[list2.index(min(list2, key = abs))]
            place_down = place[list2.index(max(list2, key = abs))]
            print(place_up, place_down, place_left, place_right)
            pts1 = np.float32([place_up, place_down, place_left, place_right])
            pts2 = np.float32([[400, 0], [400, 800], [0, 400], [800, 400]])
            print(pts1)
            print(pts2)
            M = cv.getPerspectiveTransform(pts1, pts2)

            dst = cv.warpPerspective(img, M, (800, 800))

        # print([place_up, place_down, place_left, place_right])
# Create a black image, a window and bind the function to window
img = cv.imread('oo/14.jpg')
cv.namedWindow('image')
cv.setMouseCallback('image',draw_circle)
rows,cols,ch = img.shape
place = []

cv.imshow('image',img)
cv.waitKey(0)

cv.imwrite('120.jpg',dst,)

cv.destroyAllWindows()
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
