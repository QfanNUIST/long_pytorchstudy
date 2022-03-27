"""
卷积：模拟人眼，一次感知一个patch[3x3],[5x5]
weight sharing 一个channel用同一个 kennel去取特征

优点1：减少参数，如果使用全连接，以28x28的图片为例
后面一层的feature-map的一个pixel，与上一层有28x28 的连接参数
使用 3x3 kernel feature-map的一个pixel 与前一层只有 9 个连接参数

一个kernel表征图片的一个特征，edge，blur。。。。

概念：input-channel 灰度：1 RGB：3
kernel-channel: 有多少个卷积核
kernel-size：[3x3]
stride : kernel一次移动的步长
padding ：四周补零，padding = 4

"""

'''
实际例子
train_data : [batch_number ,input_channel ,W ,H] = [b,3,28,28]

kernel : [kernel_num, input_channel ,3 ,3] = [16,3,3,3]
kernel_num 表示图片提取多少角度的特征，edge，blur，face，color。。。。。
input_channel 要与输入图像对应，一个kernel运算将，会在图片的三个通道同时滑动，得到值之后将三个计算结果加起来，得到一个kernel的值
即一个kernel_num只能得到一个feature-map,channel = 1,size = [26,26](没padding)

out：[batch_num,kernel_num,W,H] = [b,16,26,26]

'''

import torch

'''
torch.nn.Conv2d(input_channel ,kernel_num ,kernel_size ,stride ,padding)
'''
layer = torch.nn.Conv2d(1,3,kernel_size=5,stride=1,padding=2)

train_data = torch.rand(1,1,28,28)

feature = layer.forward(train_data)
print(feature.shape)
layer.weight
