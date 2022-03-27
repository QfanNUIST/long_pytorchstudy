'''
datatype     dtype          cpu-tensor          Gpu-tensor
32-bite-float torch.float    torch.FloatTensor   torch.cuda.FloatTensor
64-bite-float torch.float    torch.DoubleTensor   torch.cuda.doubleTensor
8-bit integer torch.uint8    torch.ByteTensor     torch.cuda.ByteTensor
(unsigned)
32-bite-int    torch.int     torch.IntTensor   torch.cuda.IntTensor
64-bite-int    torch.long    torch.LongTensor   torch.cuda.LongTensor

torch.tensor([list]) #list ===>tensor
torch.from_numpy(numpy)
'''

import torch

'''初始列子'''
a = torch.randn(2, 3)
print(type(a))
print(isinstance(a, torch.FloatTensor))  # 合法化检验，isinstance(a,type)查看a的类型是否与type相同

'''没有把数据送到GPU时，数据类型还不是Gputensor的类型'''
print(isinstance(a, torch.cuda.FloatTensor))
a = a.cuda()
print(isinstance(a, torch.cuda.FloatTensor))

'''tensor中标量的表示，其dimension为0'''
b = torch.tensor(1.0)

'''查看tensor形状的方法'''
a.shape

len(a)
len(a.data)

a.size()

'''dim = 1的tensor'''
'''方式一'''
a = torch.tensor([1.])
b = torch.tensor(1.)
# 加了方括号即为dim=1，没有dim=0
'''方式二'''
c = torch.FloatTensor(1)  # 虽然是圆括号，但是调用了torch.FloatTensor函数随机分配了一个一维的tensor
d = torch.FloatTensor(2)
'''方式三使用numpy转化过来'''
import numpy as np

a = np.array([1, 2.])
b = torch.from_numpy(a)
