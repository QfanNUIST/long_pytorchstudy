'''
batch_normalization
需求：
1、sigmoid函数在输入过大或者过小是会出现梯度为0，使用B_N可以将输入约束在梯度较大的范围

测试时要使用上一次的avg，var而不是本次样本的
和drop_out一样需要进行test模式切换
'''

import torch
import torch.nn as nn

'''batch_normlization函数'''
x = torch.rand(4,3,28,28)
# print(x)

layer = nn.BatchNorm2d(3)

out = layer(x)
print(layer.running_mean)
print(out)

'''batch_normlization函数'''
x = torch.rand(4,3,28*28)
print(x)

layer = nn.BatchNorm1d(3)

out = layer(x)
print(layer.running_mean)
print(out)


'''
测试时要使用上一次的avg，var而不是本次样本的
和drop_out一样需要进行test模式切换
'''
layer.eval()

test_result = layer(test_data)