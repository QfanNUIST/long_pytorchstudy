'''
建立tensor时注意torch.tensor和torch.Tensor的区别
两者都可以承载列表数据将其值转化为tensor类型torch.tensor([2,3]),torch.Tensor([2,3])
但是torch.tensor(2,3)会报错及不能生成（2，3）形状的tensor（torch.tensor(3)可以生成维度为0的tensor）
而torch.Tensor(2,3)可以生成随机数值的（2，3）形状的tensor

torch.set_default_tensor_type(torch.DoubleTensor)改变默认的tensor数据类型

torch.full   torch.randperm

arrange()和linspace的区别都指定[start,end]，但是arange最后一维表示步长，linspace表示元素总数
a = torch.randperm(10)
'''

import torch
import numpy as np

'''从numpy中获得tensor'''
a = np.array([2, 3.3])
a = torch.from_numpy(a)  # 将numpy转为tensor会自动变为torch.double及torch.float64
print(a)

b = np.ones([2, 3])
b = torch.from_numpy(b)
print(b)

'''使用list承载'''
c = torch.tensor([1, 2])
print(c)

'''使用未初始化的参数，注意将其赋值，否则容易出现torch.nan或者torch.inf'''
d = torch.empty(2, 3)
d = torch.Tensor(2, 3)
'''建议使用下面赋值的初始化'''
d = torch.ones(2, 3)
d = torch.randn(2, 3)  # 均值为0方差为1的正态分布随机数

'''设置tensor的数据类型'''
torch.tensor([1.2, 3]).type()
torch.set_default_tensor_type(torch.DoubleTensor)
torch.tensor([1.2, 3]).type()

'''torch中常用的初始化函数'''
a = torch.rand(3, 3)  # （0，1）均匀分布的3*3的tensor
b = torch.rand_like(a)  # 相当于将a.shape输入给rand()函数

c = torch.randint(0, 10, [3, 3])  # 生成[0,10)的3*3的tensor.Int的tensor
d = torch.randn(3, 3)  # 均值为0方差为1的正态分布随机数

'''full函数的应用'''
a = torch.full([2, 3], 7)  # 生成全是7的tensor

b = torch.full([], 7)  # 生成dim=0值为7的tensor
c = torch.full([1], 7)  # dim = 1

'''生成[a,b）差为c的等差数列'''
a = torch.arange(0, 10, step=2)
'''生成[a,b]一共n个的等差数列'''
b = torch.linspace(0, 10, steps=11)
c = torch.linspace(0, 10, 5)

'''随机种子shuffle torch.randperm(n, *, generator=None, out=None, 
dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False, pin_memory=False) 
→ Tensor'''
a = torch.randperm(10)
'''应用如N-Batch'''
train_data = torch.randn(4, 5)
train_label = torch.rand(4, 1)
index = torch.randperm(4)

train_data_shuffle = train_data[index]
train_label_shuffle = train_label[index]
