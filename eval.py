# -- coding: utf-8 --
"""
@Time：2023-05-19 21:51
@Author：zstar
@File：eval.py
@Describe：用于评估分割效果相关指标，本示例仅做单图评估，更多图片可类似拓展
"""
import cv2
import numpy as np
import torch
import warnings
from torch.autograd import Variable as V
from framework import MyFrame
from loss import dice_bce_loss
from networks.dlinknet import DinkNet34

warnings.filterwarnings("ignore")


class IOUMetric:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def evaluate(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            assert len(lp.flatten()) == len(lt.flatten())
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())
        # miou
        iou = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        miou = np.nanmean(iou)
        # mean acc
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.nanmean(np.diag(self.hist) / self.hist.sum(axis=1))
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
        return acc, acc_cls, iou, miou, fwavacc


if __name__ == '__main__':
    labels = []
    predicts = []
    img_path = "eval_example/104_sat.jpg"
    label_path = "eval_example/104_mask.png"

    # 加载模型
    solver = MyFrame(DinkNet34, dice_bce_loss, 2e-4)
    solver.load("weights/new.pt")

    # 读取图片，分割
    img = cv2.imread(img_path)
    img = img[None, ...].transpose(0, 3, 1, 2)
    img = V(torch.Tensor(np.array(img, np.float32) / 255.0 * 3.2 - 1.6).cuda())
    predict = solver.test_one_img(img)
    predict = np.array(predict, np.int64)

    # 读取label，二值化处理
    label = cv2.imread(label_path, 0)
    label[label > 0] = 1

    # 添加进评估列表，更多图片同理
    predicts.append(predict)
    labels.append(label)

    # 评估
    el = IOUMetric()
    acc, acc_cls, iou, miou, fwavacc = el.evaluate(predicts, labels)
    print('acc: ', acc)
    print('acc_cls: ', acc_cls)
    print('iou: ', iou)
    print('miou: ', miou)
    print('fwavacc: ', fwavacc)
