'''
index_select(dim,tensor_index)在目标数据的第dim维度的tensor_index中索引
index = torch.randperm(4)
a.index_select(0,index) 将atensor在batch_index上随机打散选择

特殊的符号 ... 表示任意多的维度，可以让计算机自己推测
a[1,...,::2].shape #第一个图像的所有通道，所有H，W每隔两个下采样
a[...,::2,::2].shape#将图片HW下采样为原来的1/2
'''
import torch
import numpy as np
import matplotlib.pyplot as plt

'''索引，以卷积神经网络的输入为例，（batch,C,H,W）'''
a = torch.rand(4, 3, 28, 28)
a[0].shape  # 第0个batch的数据
a[0, 0].shape  # 第0个batch的第0个通道的数据

'''高阶索引
a[start:stop]===>[start,stop)
a[start:]    ===>[start,end]
a[start:stop:step] ===> 在[start,end)中以step为公差索引
'''
a[0:2, :, :, :].shape  # 0-->2不包括2的batch中所有的图片
a[-1:, 1, :, :].shape  # 最后一个batch到最后一个batch  中第1个（G）通道的图片数据
a[-4:, :, :, :].shape  # 倒数第4个batch到最后一个batch  中所有通道的图片数据

'''对图片下采样'''
a[:, :, 0:-1:2, 0:-1:2].shape
a[:, :, ::2, ::2].shape

'''对tensor的某一维度采样 index_select(dim_target,[start,stop])'''
a.index_select(0, torch.tensor([0, 2])).shape

'''index_select(dim,tensor_index)在目标数据的第dim维度的tensor_index中索引'''
index = torch.randperm(4)
a.index_select(0, index)

'''特殊的符号 ... 表示任意多的维度，可以让计算机自己推测'''
a[...].shape
a[0, ...].shape

a[1, ..., ::2].shape  # 第一个图像的所有通道，所有H，W每隔两个下采样
a[..., ::2, ::2].shape  # 将图片HW下采样为原来的1/2

'''masked_select函数'''
