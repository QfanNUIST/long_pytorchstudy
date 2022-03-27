'''
MSE loss = troch.norm(label - pred,2).pow(2)

softmax 函数：S(yi) = exp(yi)/(sum(exp(y)))把大的放大更大
'''

import torch
import torch.nn.functional as F

x = torch.rand(3, 4)  # 3个数据，4个特征
w = torch.rand(4, 1,requires_grad=True)  # 4个特征输出一个结果
b = torch.rand(1,requires_grad=True)
'''
w = torch.randn(4, 1)  # 4个特征输出一个结果
w.requires_grad_()
b = torch.rand(1)
b.requires_grad_()
'''
pred = x@w +b
label = torch.rand(3,1)
loss = F.mse_loss(label,pred)

# loss.backward()
#
# print(w.grad)
# print(b.grad)

p = F.softmax(pred,dim=0)
p[1].backward() #使用反向传播的数必须为一个[1]的量。[2],[1,2]都不可以
