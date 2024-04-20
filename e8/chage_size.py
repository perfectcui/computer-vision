import cv2 as cv


def change_img_size(filename, size):
    img = cv.imread(filename)
    img = cv.resize(img, size)
    cv.imwrite(filename, img)


if __name__ == '__main__':
    f1 = 'imgs/1.jpg'
    f2 = 'imgs/2.jpg'
    size1 = (800, 600)
    size2 = (300, 400)
    change_img_size(f1, size1)
    change_img_size(f2, size2)
