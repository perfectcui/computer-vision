import copy

import cv2 as cv
import numpy as np


# SIFT特征检测函数
def detect_sift(img):
    sift = cv.xfeatures2d.SIFT_create()  # SIFT特征提取对象
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转为灰度图
    # kp = sift.detect(gray, None)  # 关键点位置
    kp, des = sift.detectAndCompute(gray, None)  # des为特征向量
    return kp, des


# 获取
def get_transform(kp1, kp2, des1, des2):
    # 使用FLANN进行匹配
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 应用比例测试
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    # 根据距离排序
    good = sorted(good, key=lambda x: x.distance)

    query_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

    # 训练图像的关键点位置
    train_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(query_pts, train_pts, cv.RANSAC, 3)

    return M


# 从src向tar做透视变换
def change_img(src_img, tar_img):
    # 首先为图像进行边界扩充
    img_extend1 = cv.copyMakeBorder(src_img, 30, 30, 30, 30, borderType=cv.BORDER_CONSTANT, value=[0, 0, 0])
    img_extend2 = cv.copyMakeBorder(tar_img, 600, 600, 600, 600, borderType=cv.BORDER_CONSTANT, value=[0, 0, 0])
    # print(img_extend2.shape)
    kp1, dps1 = detect_sift(img_extend1)
    kp2, dps2 = detect_sift(img_extend2)
    T = get_transform(kp1, kp2, dps1, dps2)

    changed = cv.warpPerspective(img_extend1, T, (img_extend2.shape[1], img_extend2.shape[0]))
    return changed


def cv_method(img_list, file_name):
    stitcher = cv.createStitcher()
    status, res = stitcher.stitch(img_list)
    if status == 0:
        cv.imwrite(file_name, res)
    else:
        print("stitcher method error!")


def merge_image(img1, img2, img3):
    # 以img2为基准，将img1,img3贴到上面
    ans = np.array(img2)
    # 求img2中全空的地方，以及img1贴上去之后还空的地方
    zero_mask1 = np.all(img2 == [0, 0, 0], axis=-1)
    zero_mask2 = np.all(np.stack((img1 == [0, 0, 0], img2 == [0, 0, 0]), axis=-1), axis=-1)
    ans[zero_mask1] = img1[zero_mask1]
    ans[zero_mask2] = img3[zero_mask2]
    return ans


if __name__ == '__main__':
    # img 1 2 3 分别是从左到右的三张图片
    imgs = [cv.imread("1.png"), cv.imread("2.png"), cv.imread("3.png")]
    c1 = change_img(imgs[0], imgs[1])
    c2 = change_img(imgs[2], imgs[1])
    img_extend = cv.copyMakeBorder(imgs[1], 600, 600, 600, 600, borderType=cv.BORDER_CONSTANT, value=[0, 0, 0])
    result = merge_image(c1, img_extend, c2)
    cv.imwrite('my_ans.png', result)

    cv_method(imgs, "cv_ans.png")
