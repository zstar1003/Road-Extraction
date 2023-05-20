# -- coding: utf-8 --
"""
@Time：2023-05-17 15:16
@Author：zstar
@File：turn_model.py
@Describe： 此脚本用于修改官方模型的key参数 (module.firstconv.weight -> firstconv.weight)
"""
import collections

import torch

if __name__ == '__main__':
    path = '../weights/log02_nllink34.th'
    model = torch.load(path)
    new_model = collections.OrderedDict([(k[7:], v) if k[:7] == 'module.' else (k, v) for k, v in model.items()])
    torch.save(new_model, "../weights/nllinknet.pt")
    # print(model)
    # print("Done")
    # print(key[0][7:])
    # key[0] = key[0][:]
    # model_key = key[0]
    # print(key[0])