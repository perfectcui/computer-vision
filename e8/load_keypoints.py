import numpy as np
import cv2 as cv
from e7.feature_mature import detect_sift, sift_show


# 输入
def construct_keypoint(the_data):  # 将numpy数组转化为keypoints数组
    keypoints = []
    for point in the_data:
        x, y, scale = point
        keypoint = cv.KeyPoint(x=x, y=y, _size=scale, _angle=0, _response=1, _octave=0, _class_id=-1)
        keypoints.append(keypoint)
    return keypoints


def load_data(filename):  # 载入参数数据
    data = np.load(filename)
    return construct_keypoint(data['keypoints']), data['descriptors']


def show_points(img, kp):
    ans = img.copy()
    cv.drawKeypoints(img, kp, ans)
    return ans


def show_match(put1, p1, d1, put2, p2, d2):
    bf = cv.BFMatcher(crossCheck=True)  # 匹配对象
    matches = bf.match(d1, d2)  # 进行两个特征矩阵的匹配
    res = cv.drawMatches(put1, p1, put2, p2, matches, None)  # 绘制匹配结果
    return res


# def select_num_of_points(num, keypoints):
#     ans = sorted(keypoints, key=lambda x: x.response, reverse=True)[:num]
#     return ans


if __name__ == '__main__':
    img1 = cv.imread('imgs/1.jpg')
    img2 = cv.imread('imgs/2.jpg')
    # kp是关键点，dps是特征向量
    kp1, dps1 = detect_sift(img1)
    kp2, dps2 = detect_sift(img2)
    rd_kp1, rd_dps1 = load_data("imgs/1.jpg.r2d2")
    rd_kp2, rd_dps2 = load_data("imgs/2.jpg.r2d2")

    # 分别展示keypoints
    sift_1 = show_points(img1, kp1)
    sift_2 = show_points(img2, kp2)
    rd_1 = show_points(img1, rd_kp1)
    rd_2 = show_points(img2, rd_kp2)
    cv.imshow('sift', sift_1)
    cv.imshow('rd', rd_1)
    cv.waitKey(0)
    cv.imshow('sift', sift_2)
    cv.imshow('rd', rd_2)
    cv.waitKey(0)

    match1 = show_match(img1, kp1, dps1, img2, kp2, dps2)
    match2 = show_match(img1, rd_kp1, rd_dps1, img2, rd_kp2, rd_dps2)
    cv.imshow('sift', match1)
    cv.imshow('rd', match2)
    cv.waitKey(0)
