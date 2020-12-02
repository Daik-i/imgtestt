import cv2 as cv
import numpy as np

# 在图上找点标出，有新模板时用
def circle_opt(img):
    cv.circle(img, center, 3, (0, 0, 255), -1)
    cv.circle(img, center, 195, (0, 0, 255), 1)
    cv.imshow("result", img)
    cv.waitKey(0)

# ORB特征匹配
def ORB_Feature(img1, img2):
    # 初始化ORB
    orb = cv.ORB_create()

    # 寻找关键点,计算描述符(特征向量)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # 初始化 BFMatcher
    bf = cv.BFMatcher(cv.NORM_HAMMING)

    # 对描述子进行匹配
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)     # 将关键点匹配信息按特征向量的距离进行升序排列

    # 筛选匹配点，选最匹配的20个匹配点
    goodMatch = matches[:20]
    # 仿射变换所需的匹配点数必须大于4
    # 将匹配点对应的关键点位置信息取出并求仿射变换矩阵
    if len(goodMatch) > 4:
        ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ransacReprojThreshold = 2
        M, status = cv.findHomography(ptsA, ptsB, cv.RANSAC, ransacReprojThreshold)
        result = cv.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]), flags=cv.INTER_LINEAR +
                                            cv.WARP_INVERSE_MAP)
        cv.imshow('image', result)
        cv.waitKey(0)
        # circle_opt(result)
        # # 绘制匹配结果
        # draw_match(img1, img2, kp1, kp2, goodMatch)
    else:
        return None
    return result

# 画出匹配点
def draw_match(img1, img2, kp1, kp2, match):
    outimage = cv.drawMatches(img1, kp1, img2, kp2, match, outImg=None)
    cv.imshow("Match Result", outimage)
    cv.waitKey(0)

# 将仿射变换后的图像进行高斯去噪、灰度化、边缘检测
def img_chuli(img):
    img_gauss = cv.GaussianBlur(img, (3, 3), 1)
    cv.imshow('image', img_gauss)
    cv.waitKey(0)
    img_gray = cv.cvtColor(img_gauss, cv.COLOR_RGB2GRAY)
    cv.imshow('image', img_gray)
    cv.waitKey(0)
    canny_edge2 = cv.Canny(img_gray, threshold1=180, threshold2=230)
    cv.imshow("image", canny_edge2)
    cv.waitKey(0)
    return canny_edge2

# 将边缘检测过的图像进行遮盖，只保留指针轨迹部分
def img_mask(img):
    roi_img = np.zeros(img.shape[0:2], dtype=np.uint8)              # 掩码需要二进制图像
    zero = img * 0
    cv.circle(roi_img, center, 165, (255, 255, 255), -1)
    img_add_mask = cv.add(img, zero, mask=roi_img)
    cv.circle(img_add_mask, center, 75, (0, 0, 0), -1)
    cv.imshow("image", img_add_mask)
    cv.waitKey(0)
    return img_add_mask

# 进行读值，先读左值，若无值再读右值
def img_duzhi(img):
    jieguo_zuo = detect_zuo(img)
    if jieguo_zuo:
        print(jieguo_zuo)
        print("表盘数值为：{0:<.3f}".format(jieguo_zuo))
    else:
        jieguo_you = detect_you(img)
        if jieguo_you:
            print("表盘数值为：{0:<.3f}".format(jieguo_you))
        else:
            if cuowu == 0:
                print('未检测到指针')
            else :
                print('指针检测不准确，请重新检测')
    cv.imshow("image", result1)
    cv.waitKey(0)

# 读左值前先将右侧面盖住
def detect_zuo(img):
    roi_img = np.zeros([500,500], dtype=np.uint8)  # 掩码需要二进制图像
    roi_img[0:500,0:251] = 255
    zero = img * 0
    img_add_mask = cv.add(img, zero, mask=roi_img)
    cv.imshow("image", img_add_mask)
    cv.waitKey(0)
    theta = get_theta(img_add_mask)
    if isinstance(theta,bool):
        return False
    else:
        if theta <= theta_first-1:
            print("指针超出范围")
            return False
        elif theta_first-1 < theta <= theta_second:
            zhi = (theta - theta_first) / (theta_second - theta_first) * 0.05
            return zhi
        else:
            zhi = (theta - theta_second) / (180 - theta_second) * 0.75 + 0.05
            return zhi

# 读右值前先将左侧面盖住
def detect_you(img):
    roi_img = np.zeros([500,500], dtype=np.uint8)  # 掩码需要二进制图像
    roi_img[0:500,251:] = 255
    zero = img * 0
    img_add_mask = cv.add(img, zero, mask=roi_img)
    cv.imshow("image", img_add_mask)
    cv.waitKey(0)
    theta = get_theta(img_add_mask)
    if isinstance(theta,bool):
        return False
    else:
        if theta <= theta_end+1:
            zhi = theta / theta_end * 0.8 + 0.8
            return zhi
        else:
            print("指针超出范围")
            return False

# 取得图中直线的角度，HoughLines返回的theta是直线与y轴的夹角，顺时针为正方向，范围0到pi
def get_theta(img):
    global cuowu
    lines = cv.HoughLines(img, 1, np.pi / 180, 60)
    try:
        lenth = len(lines)
    except:
        lenth = 0
    if lenth == 0 :
        return False
    elif lenth == 1:
        draw_line(lines[0][0], result1)
        theta = lines[0][0][1]
        print(theta)
        return theta*180/np.pi
    elif 2 <= lenth <= 5:
        all_theta = []
        for line in lines:
            draw_line(line[0], result1)
            all_theta.append(line[0][1])
        theta_fanwei = np.ptp(all_theta)
        if theta_fanwei < 4:
            theta = np.mean(all_theta)
            return theta*180/np.pi
        else:
            cuowu += 1
            return False
    else:
        cuowu += 1
        for line in lines:
            draw_line(line[0], result1)
        return False

# 画出检测出的直线
def draw_line(line, img):
    rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * ( a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * ( a))
    cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)        # 此函数画的是线段，故须较远的两点

if __name__ == '__main__':
    # 读取图片
    # moban = cv.imread('oo/moban.jpg')
    # circle_opt(moban)
    center = (251,255)
    cuowu = 0
    theta_first = 48.667
    theta_second = 53.800
    theta_end = 135.333
    image1 = cv.imread('oo/moban.jpg')
    image2 = cv.imread('oo/mobana.jpg')
    result = result1 = ORB_Feature(image1, image2)
    # cv.circle(result1, (100,200), 3, (255, 0, 0), -1)
    # cv.imshow("image", result1)
    # cv.waitKey(0)
    img_bian = img_chuli(result)
    img_opt = img_mask(img_bian)
    img_duzhi(img_opt)
