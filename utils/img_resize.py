# -- coding: utf-8 --
"""
@Time：2023-05-19 21:09
@Author：zstar
@File：img_resize.py
@Describe：此脚本用于批量resize图片至(1024,1024)，以满足训练图像分辨率需要
"""
import os
import cv2
from tqdm import tqdm

if __name__ == '__main__':
    datadir = "E:/dataset"
    Resize_size = 1024, 1024
    path = os.path.join(datadir)
    img_list = os.listdir(path)

    for i in tqdm(img_list):
        img_array = cv2.imread(os.path.join(path, i))
        new_array = cv2.resize(img_array, Resize_size)
        img_name = str(i)
        save_path = datadir + '/' + str(i)
        cv2.imwrite(save_path, new_array)
