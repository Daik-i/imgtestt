import cv2 as cv
import numpy as np

path_input = 'oo/001.jpg'
path_output = "001.txt"
img_rgb = cv.imread(path_input)
# cv.imshow('image',img_rgb)
# cv.waitKey(0)

try:
    img_gray = cv.cvtColor(img_rgb,cv.COLOR_RGB2GRAY)
    # cv.imshow('image',img_gray)
    # cv.waitKey(0)
except:
    img_gray = img_rgb

file = open(path_output,'a')
file.write(str(img_gray))
file.close()

# cv.imwrite('res83.pmg',img_gray)
# print(img_gray[0])

