import torch
import cv2
import os
import numpy as np
import warnings
from time import time
from pathlib import Path
from torch.autograd import Variable as V
from tqdm import tqdm
from networks.unet import Unet
from networks.dunet import Dunet
from networks.linknet import LinkNet34
from networks.dlinknet import DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool
from networks.nllinknet import NL34_LinkNet
from utils.show_img import show_img

warnings.filterwarnings("ignore")
resize_settings = 1024, 1024


# TTA:Test Time Augmentation
class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()

    def load(self, path):
        self.net.load_state_dict(torch.load(path))
        self.net.eval()

    def segment(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, resize_settings)  # Shape:(1024, 1024, 3)
        img90 = np.array(np.rot90(img))  # Shape:(1024, 1024, 3)
        img1 = np.concatenate([img[None, ...], img90[None, ...]])  # Shape:(2, 1024, 1024, 3) img[None]:增加第一个位置维度
        img2 = np.array(img1)[:, ::-1]  # 垂直翻转
        img3 = np.array(img1)[:, :, ::-1]  # 水平翻转
        img4 = np.array(img2)[:, :, ::-1]  # 垂直翻转+水平翻转
        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        # 将图像像素值标准化至[-1.6,1.6]
        # 原因参考：https://github.com/zlckanata/DeepGlobe-Road-Extraction-Challenge/issues/8
        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()  # img1:Shape:(2, 1, 1024, 1024) -> (2, 1024, 1024)
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]
        # show_img(maska[0], mask2)  # 对比单图检测和TTA之后的效果
        return mask2


if __name__ == '__main__':
    source = 'test_example/'
    val = os.listdir(source)
    solver = TTAFrame(DinkNet34)
    solver.load('weights/dlinknet.pt')
    tic = time()
    target = 'results/'
    if not Path(target).exists():
        os.mkdir(target)
    for i, name in tqdm(enumerate(val)):
        mask = solver.segment(source + name)
        mask[mask > 4.0] = 255
        mask[mask <= 4.0] = 0
        mask = np.concatenate([mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=2)
        cv2.imwrite(target + name[:-4] + '_mask.png', mask.astype(np.uint8))
    print("Done")
