'''

'''

import torch
import torch.nn.functional as F


'''体验pooling'''
x = torch.rand(1,1,14,14)

# layer = torch.nn.MaxPool2d(2,stride = 2)
layer = torch.nn.AvgPool2d(2,stride = 2)
out = layer(x)
print(out.shape)

'''up-sampling
F.interpolate(tensor,scale_factor=n,mode='')
对目标tensor进行up-sampling，放大scare_factor=n倍，以mode=''放大，常用nearest,双线插值，三线插值

'''
# x= torch.rand(1,1,7,7)
out = F.interpolate(out,scale_factor=2,mode = 'nearest')
print(out.shape)

'''relu
将负数变为零
'''
layer = torch.nn.ReLU(inplace=True)
out = layer(out)
print(out.shape)
print(out)

