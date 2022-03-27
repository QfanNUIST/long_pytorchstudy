'''
torch中的加减乘除运算
a-b  == sub(a,b)
a+b == add(a,b)

* 是点乘，对应元素相乘
@ = torch.matmul(a,b) 是矩阵乘法运算,

a.floor()  # 向小取整
a.ceil()  # 向大取整
a.trunc()  # 取整数部分
a.frac()  # 取小数部分
a.round()  # 四舍五入

clamp（min,max）常用于将W的梯度限制在10以内
'''

import torch

'''2d中的点乘和矩阵乘法'''
a = torch.ones(2, 2)
b = torch.full([2, 2], 3)

c = a * b  # [[3,3],[3,3]]
d1 = torch.matmul(a, b)
d2 = a @ b

'''nD中的矩阵乘法
只有最小的两维进行矩阵乘法运算，要求对应维度size正确
'''
a = torch.rand(4, 3, 28, 14)
w1 = torch.rand(4, 3, 28, 14)
a1 = torch.matmul(a, w1.transpose(2, 3))  # [4,3,28,28]

w2 = torch.rand(4, 1, 14, 28)  # 广播
a2 = torch.matmul(a, w2)

'''矩阵次方tensor.pow(num)
tensor**num
'''
a = torch.full([3, 3], 9)  # 3*3矩阵值为9
b1 = a ** (1 / 2)
b2 = a.sqrt()  # 对a矩阵开根号

c1 = b1 ** (2)
c2 = b1.pow(2)

'''torch 中的指数函数和对数函数'''
a = torch.exp(torch.full([2, 2], 1))
b = torch.log(a)
c = torch.log2(a)

'''torch中的向上向下取整，保留整数小数'''
a = torch.tensor([3.14])

a.floor()  # 向小取整
a.ceil()  # 向大取整
a.trunc()  # 取整数部分
a.frac()  # 取小数部分

a.round()  # 四舍五入

'''矩阵元素限幅 clamp（min,max）常用于将W的梯度限制在10以内'''
a = torch.rand(3, 3) * 15
b = a.clamp(10)  # 最小值设为10，如果a的元素有小于10的转化为10
c = a.clamp(0, 10)  # tensor(a)中元素的值限制在[0,10]之间
