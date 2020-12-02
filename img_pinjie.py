import cv2 as cv
import numpy as np

def ORB_Feature(img1, img2):
    # 初始化ORB
    orb = cv.ORB_create()

    # 寻找关键点
    kp1 = orb.detect(img1)
    kp2 = orb.detect(img2)

    # 计算描述符
    kp1, des1 = orb.compute(img1, kp1)
    kp2, des2 = orb.compute(img2, kp2)

    # 初始化 BFMatcher
    bf = cv.BFMatcher(cv.NORM_HAMMING)

    # 对描述子进行匹配
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)     # 将关键点匹配信息按特征向量的距离进行升序排列

    # 筛选匹配点，选最匹配的20个匹配点
    goodMatch = matches[:40]
    # 仿射变换所需的匹配点数必须大于4
    # 将匹配点对应的关键点位置信息取出并求仿射变换矩阵
    if len(goodMatch) > 4:
        ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ransacReprojThreshold = 3.3          # 同一数值在不同版本中效果不同，例如在4.4中最优的4也没有3.4.1.15中的最优2效果好
        M, status = cv.findHomography(ptsA, ptsB, cv.RANSAC, ransacReprojThreshold)

        # 将仿射的目的图片设置为两张图片的宽度大小，因为向右拼接的图片仿射就以交界的坐标为基准，故放射后就自然在拼接处
        # 然后将左图复杂到左侧区域，自然盖住右图延申过来的区域
        result = cv.warpPerspective(img2, M, (img1.shape[1] + img2.shape[1], img1.shape[0]),flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
        result[0:img2.shape[0], 0:img1.shape[1]] = img1
        cv.imshow('image', result)
        cv.waitKey(0)

        # 绘制匹配结果
        draw_match(img1, img2, kp1, kp2, goodMatch)
    else:
        return None

def draw_match(img1, img2, kp1, kp2, match):
    outimage = cv.drawMatches(img1, kp1, img2, kp2, match, outImg=None)
    cv.imshow("Match Result", outimage)
    cv.waitKey(0)


if __name__ == '__main__':
    # 读取图片
    image1 = cv.imread('oo/Q1.jpg')
    image2 = cv.imread('oo/Q2.jpg')
    ORB_Feature(image1, image2)