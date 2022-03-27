'''
y = xw + b

'''

import torch
import torch.nn.functional as F

train_data = torch.rand(4,10)
train_label = torch.rand(4,1).expand(4,5)
w = torch.rand(10,5,requires_grad=True) #10个特征，5个分类结果
b = torch.rand(5,requires_grad=True)

pre = train_data@w + b
pre = torch.relu(pre)
loss = F.mse_loss(pre,train_label)

loss.backward()

print(w.grad)