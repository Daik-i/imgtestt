import cv2 as cv
import numpy as np

def circle_opt(img):
    cv.circle(img, center, 3, (0, 0, 255), -1)
    cv.circle(img, center, 195, (0, 0, 255), 1)
    cv.imshow("result", img)
    cv.waitKey(0)


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

        # 将仿射的目的图片设置为两张图片的宽度大小，因为向右拼接的图片仿射就以交界的坐标为基准，故放射后就自然在拼接处
        # 然后将左图复杂到左侧区域，自然盖住右图延申过来的区域
        result = cv.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]), flags=cv.INTER_LINEAR +
                                            cv.WARP_INVERSE_MAP)
        # result[0:img2.shape[0], 0:img1.shape[1]] = img1
        cv.imshow('image', result)
        cv.waitKey(0)
        # circle_opt(result)

        # # 绘制匹配结果
        # draw_match(img1, img2, kp1, kp2, goodMatch)
    else:
        return None
    return result

def draw_match(img1, img2, kp1, kp2, match):
    outimage = cv.drawMatches(img1, kp1, img2, kp2, match, outImg=None)
    cv.imshow("Match Result", outimage)
    cv.waitKey(0)

def img_chuli(img):
    img_gauss = cv.GaussianBlur(img, (3, 3), 1)
    cv.imshow('image', img_gauss)
    cv.waitKey(0)
    img_gray = cv.cvtColor(img_gauss, cv.COLOR_RGB2GRAY)
    cv.imshow('image', img_gray)
    cv.waitKey(0)
    # thre1 = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 7, 2)
    # # ret, thre1 = cv.threshold(img_gray, 125, 255, cv.THRESH_BINARY)
    # cv.imshow('image', thre1)
    # cv.waitKey(0)
    canny_edge2 = cv.Canny(img_gray, threshold1=180, threshold2=230)
    cv.imshow("image", canny_edge2)
    cv.waitKey(0)
    return canny_edge2

def img_mask(img):
    roi_img = np.zeros(img.shape[0:2], dtype=np.uint8)              # 掩码需要二进制图像
    zero = img * 0
    cv.circle(roi_img, center, 145, (255, 255, 255), -1)
    img_add_mask = cv.add(img, zero, mask=roi_img)
    cv.circle(img_add_mask, center, 75, (0, 0, 0), -1)
    cv.imshow("image", img_add_mask)
    cv.waitKey(0)
    return img_add_mask



def img_duzhi(img):

    # cv.circle(canny_edge2, center, 75, (0, 0, 0), 3)
    # cv.circle(canny_edge2, center, 145, (0, 0, 0), 3)
    # cv.imshow("result", canny_edge2)
    # cv.waitKey(0)


    line_K = []
    line_b = []
    lines = cv.HoughLines(img, 1, np.pi / 180, 40)
    print(lines)
    if len(lines) == 0:
        print('未检测到指针')
    elif len(lines) == 1:
        x1, y1, x2, y2, rho, theta = get_line_point(lines[0][0], result1)
        line_theta = 90 - np.degrees(np.arctan((center[1] - y1) / (x1 - center[0])))  # 度数需要再考虑一下
        print(line_theta)
        print('请选择初始位置和终止位置')
    elif len(lines) == 2:
        for line in lines:
            x1, y1, x2, y2, rho, theta = get_line_point(line[0], result1)
            A = [x1, x2]
            B = [y1, y2]
            kb = get_cross_point(A, [1, 1], B)
            line_K.append(kb[0])
            line_b.append(kb[1])
        xy = get_cross_point([line_K[0] * -1, line_K[1] * -1], [1, 1], line_b)
        cv.circle(img, (int(xy[0]), int(xy[1])), 3, (255, 0, 0), -1)
        if center[0] <= xy[0]:
            line_theta = 90 - np.degrees(np.arctan((center[1] - xy[1]) / (xy[0] - center[0])))
        else:
            line_theta = -90 - np.degrees(np.arctan((center[1] - xy[1]) / (xy[0] - center[0])))
        print(line_theta)
        print('请选择初始位置和终止位置')
    else:
        for line in lines:
            x1, y1, x2, y2, rho, theta = get_line_point(line[0], result1)
        print('指针检测不准确，请重新检测')
    cv.imshow("image", result1)
    cv.waitKey(0)
    # print((line_theta-start_theta),(end_theta-start_theta))
    # res_dst = (line_theta-start_theta)/(end_theta-start_theta)*1.6
    # print(res_dst)
    # cv.imshow("image", img_rgb)
    # cv.waitKey(0)



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
    return x1, y1, x2, y2, rho, theta

# 给出线性方程组并求解
def get_cross_point(x,y,z):
    A = np.array([x, y], dtype = float)
    B = np.array(z)
    res_m = np.linalg.solve(A.T, B)
    print(res_m)
    return res_m

if __name__ == '__main__':
    # 读取图片
    # moban = cv.imread('oo/moban.jpg')
    # circle_opt(moban)
    center = (251,255)
    image1 = cv.imread('oo/moban.jpg')
    image2 = cv.imread('oo/2224.jpg')
    result = result1 = ORB_Feature(image1, image2)
    img_bian = img_chuli(result)
    img_opt = img_mask(img_bian)
    img_duzhi(img_opt)