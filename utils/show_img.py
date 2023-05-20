# -- coding: utf-8 --
"""
@Time：2023-05-19 19:37
@Author：zstar
@File：show_img.py
@Describe：此函数用于并排显示图像，调试用
"""
"""
读取、保存、显示图像
"""
import cv2
import numpy as np


def show_img(img1, img2):
    img1 = cv2.resize(img1, (500, 500))
    img2 = cv2.resize(img2, (500, 500))
    imgStackH = np.hstack((img1, img2))
    cv2.imshow("img_show", imgStackH)
    cv2.waitKey(0)


if __name__ == '__main__':
    img_path = "../test_example/2667_sat.jpg"
    img1 = cv2.imread(img_path)
    img2 = cv2.imread(img_path)
    show_img(img1, img2)
