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
    # print(kp1[0],des1[0])
    # 画出关键点
    # outimg1 = cv.drawKeypoints(img1, keypoints=kp1, outImage=None)
    # outimg2 = cv.drawKeypoints(img2, keypoints=kp2, outImage=None)
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
            print(x.queryIdx,x.trainIdx,x.imgIdx)

    # 将匹配好的点的坐标信息提取出来
    point_first = []
    point_second = []
    for i in good_match:
        id_first = i.queryIdx
        id_second = i.trainIdx
        point_first.append(list(kp1[id_first].pt))
        point_second.append(list(kp1[id_second].pt))
    print(point_first)

    # 找出其中位置较好的点
    np_first = np.array(point_first)
    np_second = np.array(point_second)
    small1_id = np.argmin(np_first, 0)          # 最左边点和最上边点的索引
    big1_id = np.argmax(np_first, 0)            # 最右边和最上边点的索引
    small2_id = np.argmin(np_second, 0)
    big2_id = np.argmax(np_second, 0)
    place_left1 = point_first[small1_id[0]]
    place_right1 = point_first[big1_id[0]]
    place_up1 = point_first[small1_id[1]]
    place_down1 = point_first[big1_id[1]]
    place_left2 = point_second[small1_id[0]]
    place_right2 = point_second[big1_id[0]]
    place_up2 = point_second[small1_id[1]]
    place_down2 = point_second[big1_id[1]]
    pts1 = np.float32([place_left1,place_right1,place_up1,place_down1])
    pts2 = np.float32([place_left2, place_right2, place_up2, place_down2])
    # print(small1_id)
    # print(big1_id)
    # print(small2_id)
    # print(big2_id)
    cv.circle(img2, (int(place_left2[0]), int(place_left2[1])), 8, (255, 0, 0), -1)
    cv.circle(img2, (int(place_right2[0]), int(place_right2[1])), 8, (255, 0, 0), -1)
    cv.circle(img2, (int(place_up2[0]), int(place_up2[1])), 8, (255, 0, 0), -1)
    cv.circle(img2, (int(place_down2[0]), int(place_down2[1])), 8, (255, 0, 0), -1)
    cv.circle(img1, (int(place_left1[0]), int(place_left1[1])), 8, (255, 0, 0), -1)
    cv.circle(img1, (int(place_right1[0]), int(place_right1[1])), 8, (255, 0, 0), -1)
    cv.circle(img1, (int(place_up1[0]), int(place_up1[1])), 8, (255, 0, 0), -1)
    cv.circle(img1, (int(place_down1[0]), int(place_down1[1])), 8, (255, 0, 0), -1)


    cv.imshow('image2', img2)
    cv.waitKey(0)
    cv.imshow('image1', img1)
    cv.waitKey(0)

    # 仿射变换
    M = cv.getPerspectiveTransform(pts1, pts2)

    dst = cv.warpPerspective(img2, M, (800, 800))

    cv.imshow('image', dst)
    cv.waitKey(0)

    # 绘制匹配结果
    draw_match(img1, img2, kp1, kp2, good_match)


def draw_match(img1, img2, kp1, kp2, match):
    outimage = cv.drawMatches(img1, kp1, img2, kp2, match, outImg=None)
    cv.imshow("Match Result", outimage)
    cv.waitKey(0)


if __name__ == '__main__':
    # 读取图片
    image1 = cv.imread('oo/222.jpg')
    image2 = cv.imread('oo/2224.jpg')
    ORB_Feature(image1, image2)