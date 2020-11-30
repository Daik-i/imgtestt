import cv2 as cv


def ORB_Feature(img1, img2):
    # 初始化ORB
    orb = cv.ORB_create()

    # 寻找关键点
    kp1 = orb.detect(img1)
    kp2 = orb.detect(img2)

    # 计算描述符
    kp1, des1 = orb.compute(img1, kp1)
    kp2, des2 = orb.compute(img2, kp2)

    '''
    ORB.detect(img1)和ORB.compute(img1, kp1)可同时使用，即 kp1, des1 = orb.detectAndCompute(img1,None)
    kp1     是关键点的信息，其中元素的pt属性即为位置，如  kp1[0].pt  pt指的是元组 tuple(x,y)
    des1    是关键点的特征构成的32维向量
    '''

    # 画出关键点
    outimg1 = cv.drawKeypoints(img1, keypoints=kp1, outImage=None)
    outimg2 = cv.drawKeypoints(img2, keypoints=kp2, outImage=None)

    # 显示关键点   #将两幅图拼接在一起
    import numpy as np
    outimg3 = np.hstack([outimg1, outimg2])
    print(outimg3)
    cv.imshow("Key Points", outimg3)
    cv.waitKey(0)

    # 初始化 BFMatcher
    bf = cv.BFMatcher(cv.NORM_HAMMING)

    # 对描述子进行匹配
    matches = bf.match(des1, des2)

    '''
    matches中保存的有两图对应关键点的匹配信息
    matches[i].distance     是两个关键点的各自特征组成的32维向量的距离，即特征匹配程度
    matches[i].queryIdx     猜测是关键点在前一幅图中的索引
    matches[i].trainIdx     猜测是关键点在后一幅图中的索引
    matches[i].imgIdx       未知
    '''
    # 计算最大距离和最小距离
    min_distance = matches[0].distance
    max_distance = matches[0].distance

    for x in matches:
        if x.distance < min_distance:
            min_distance = x.distance
        if x.distance > max_distance:
            max_distance = x.distance

    print(matches[0].distance,matches[1].imgIdx,matches[5].queryIdx,matches[5].trainIdx)


    # 筛选匹配点
    '''
        当描述子之间的距离大于两倍的最小距离时，认为匹配有误。
        但有时候最小距离会非常小，所以设置一个经验值30作为下限。
    '''
    good_match = []
    for x in matches:
        if x.distance <= max(1.8* min_distance, 30):
            good_match.append(x)

    # 绘制匹配结果
    draw_match(img1, img2, kp1, kp2, good_match)


def draw_match(img1, img2, kp1, kp2, match):
    outimage = cv.drawMatches(img1, kp1, img2, kp2, match, outImg=None)
    # cv.imshow("Match Result", outimage)
    # cv.waitKey(0)


if __name__ == '__main__':
    # 读取图片
    image1 = cv.imread('oo/2224.jpg')
    image2 = cv.imread('oo/222.jpg')
    ORB_Feature(image1, image2)