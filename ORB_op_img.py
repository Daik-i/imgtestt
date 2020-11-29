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
    # print(kp1[0],des1[0])
    # 画出关键点
    outimg1 = cv.drawKeypoints(img1, keypoints=kp1, outImage=None)
    outimg2 = cv.drawKeypoints(img2, keypoints=kp2, outImage=None)
    # print(outimg1)
    # cv.imshow("Key Points", outimg1)
    # cv.waitKey(0)
    # 显示关键点   #将两幅图拼接在一起
    # import numpy as np
    # outimg3 = np.hstack([outimg1, outimg2])
    # print(outimg3)
    # cv.imshow("Key Points", outimg3)
    # cv.waitKey(0)
    print(kp1[0].pt)
    # 初始化 BFMatcher
    bf = cv.BFMatcher(cv.NORM_HAMMING)

    # 对描述子进行匹配
    matches = bf.match(des1, des2)
    print(len(matches))
    # 计算最大距离和最小距离
    min_distance = matches[0].distance
    max_distance = matches[0].distance

    for x in matches:
        if x.distance < min_distance:
            min_distance = x.distance
        if x.distance > max_distance:
            max_distance = x.distance
    print(min_distance,max_distance)
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
            # print(x)
            print(x.queryIdx,x.trainIdx,x.imgIdx)



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