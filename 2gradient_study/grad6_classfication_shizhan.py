"""
假设输入为 28*28的图片
hidden1 [ch_out,ch_in]=[200,784]
hidden2 [ch_out,ch_in] = [200,200]
hidden3 [ch_out,ch_in] = [10,200]
"""

import torch
import torch.nn.functional as F

# hidden layer1
w1 = torch.rand(200, 784, requires_grad=True)
b1 = torch.rand(1, 200, requires_grad=True)
# hidden layer2
w2 = torch.rand(200, 200, requires_grad=True)
b2 = torch.rand(1, 200, requires_grad=True)
# hidden layer1
w3 = torch.rand(10, 200, requires_grad=True)
b3 = torch.rand(1, 10, requires_grad=True)

'''定义模型'''


def forward(x):
    hidden1 = x @ w1.t() + b1
    x = F.relu(hidden1)
    hidden2 = x @ w2.t() + b2
    x = F.relu(hidden2)
    hidden3 = x @ w2.t() + b2
    y = F.relu(hidden3)
    return y


'''设置网络优化器及loss'''
optimizer = torch.optim.SGD([w1, b1, w2, b2, w3, b3], lr=1e-3)
criteon = F.cross_entropy()

'''训练模型'''
epoches = 10000

for epoch in range(epoches):
    for batch_idx,(data,target) in enumerate(train_loader):
        data = data.view(-1,28*28)

        pred = forward(data)
        loss = criteon(pred,target)