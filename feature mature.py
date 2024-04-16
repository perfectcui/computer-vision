import cv2 as cv
import time


# import matplotlib.pyplot as plt


def detect_sift(img):
    sift = cv.xfeatures2d.SIFT_create()  # SIFT特征提取对象
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转为灰度图
    kp = sift.detect(gray, None)  # 关键点位置
    kp, des = sift.detectAndCompute(gray, None)  # des为特征向量
    return kp, des


def detect_surf(img):
    # sift = cv.SIFT_create()  # SIFT特征提取对象
    surf = cv.xfeatures2d.SURF_create(400)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转为灰度图
    # kp = surf.detect(gray, None)  # 关键点位置
    kp, des = surf.detectAndCompute(gray, None)
    # kp, des = surf.compute(gray, kp)  # des为特征向量
    return kp, des


def detect_ORB(img):
    # sift = cv.SIFT_create()  # SIFT特征提取对象
    orb = cv.ORB_create()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转为灰度图
    # kp = surf.detect(gray, None)  # 关键点位置
    kp, des = orb.detectAndCompute(gray, None)
    # kp, des = surf.compute(gray, kp)  # des为特征向量
    return kp, des


def sift_show(put1, put2):
    s1 = time.clock()
    kp1, des1 = detect_sift(put1)
    kp2, des2 = detect_sift(put2)
    s2 = time.clock()
    print("sift cost {}us".format(s2 - s1))
    ans1 = put1.copy()
    ans2 = put2.copy()
    cv.drawKeypoints(put1, kp1, ans1)
    cv.drawKeypoints(put2, kp1, ans2)
    cv.imshow('shif_ans1', ans1)
    cv.imshow('shif_ans2', ans2)
    bf = cv.BFMatcher(crossCheck=True)  # 匹配对象
    matches = bf.match(des1, des2)  # 进行两个特征矩阵的匹配
    res = cv.drawMatches(put1, kp1, put2, kp2, matches, None)  # 绘制匹配结果
    cv.imshow("shif_match", res)
    cv.waitKey(0)
    cv.destroyAllWindows()


def surf_show(put1, put2):
    s1 = time.clock()
    kp1, des1 = detect_surf(put1)
    kp2, des2 = detect_surf(put2)
    s2 = time.clock()
    print("surf cost {}us".format(s2 - s1))
    ans1 = put1.copy()
    ans2 = put2.copy()
    cv.drawKeypoints(put1, kp1, ans1)
    cv.drawKeypoints(put2, kp1, ans2)
    cv.imshow('surf_ans1', ans1)
    cv.imshow('surf_ans2', ans2)
    bf = cv.BFMatcher(crossCheck=True)  # 匹配对象
    matches = bf.match(des1, des2)  # 进行两个特征矩阵的匹配
    res = cv.drawMatches(put1, kp1, put2, kp2, matches, None)  # 绘制匹配结果
    # plt.imshow(res)
    cv.imshow("surf_match", res)
    cv.waitKey(0)
    cv.destroyAllWindows()


def orb_show(put1, put2):
    s1 = time.clock()
    kp1, des1 = detect_ORB(put1)
    kp2, des2 = detect_ORB(put2)
    s2 = time.clock()
    print("orb cost {}us".format(s2 - s1))
    ans1 = put1.copy()
    ans2 = put2.copy()
    cv.drawKeypoints(put1, kp1, ans1)
    cv.drawKeypoints(put2, kp1, ans2)
    cv.imshow('orb_ans1', ans1)
    cv.imshow('orb_ans2', ans2)
    bf = cv.BFMatcher(crossCheck=True)  # 匹配对象
    matches = bf.match(des1, des2)  # 进行两个特征矩阵的匹配
    res = cv.drawMatches(put1, kp1, put2, kp2, matches, None)  # 绘制匹配结果
    # plt.imshow(res)
    cv.imshow("orb_match", res)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    img1 = cv.imread('1.jpg')
    img2 = cv.imread('2.jpg')
    img1 = cv.resize(img1, (800, 600))
    img2 = cv.resize(img2, (300, 400))
    sift_show(img1, img2)
    surf_show(img1, img2)
    orb_show(img1, img2)
