'''使用神经网络完成对手写字的识别'''
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import matplotlib.pyplot as plt

from utils import plot_image, plot_curve, one_hot

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

batch_size = 512

'''load dataset'''
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=False, download=True,
                               # 加载mnist数据集，路径为‘mnist_data’，加载的mnist数据集中的train_set，如果没有将从网上下载
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)

# x,y = next(iter(train_loader))
# print(x.shape,y.shape,x.min(),x.max())
# plot_image(x,y,'image sample')

'''模型搭建'''


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.h1 = nn.Linear(28 * 28, 256)
        self.h2 = nn.Linear(256, 64)
        self.h3 = nn.Linear(64, 10)
        # Mnist手写字识别只有十种类别

    def forward(self, x):
        x = F.relu(self.h1(x))
        # x [b,1,28,28]
        # h1 = xw + b
        x = F.relu(self.h2(x))
        # h2 = h1*w + b
        x = self.h3(x)
        return x


device = torch.device('cuda:0')
net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.01)
# certion = F.mse_loss
epoches = 3
loss_save = []
for epoch in range(epoches):
    for index, (x, label) in enumerate(train_loader):
        '''由于线性层的输入为1*28*28的tensor，所以我们要将
        [batch_size,1,28,28]的手写图像数据转化
        [batch_size,1,28*28]的向量
        '''
        x = x.view(x.shape[0], 28 * 28)

        pre = net(x)
        # label = one_hot(label, 10)

        loss = F.cross_entropy(pre, label)
        loss_save.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if index % 10 == 0:
            print(epoch, index, loss.item())

plot_curve(loss_save)  # 可视化loss

'''测试集准确度编程'''
total_cor = 0
total_num = 0
for index, (x, label_test) in enumerate(test_loader):
    x = x.view(x.shape[0], 28 * 28)
    out = net(x)
    # 将[batch_size，10]的out变为[batch_size,1]的输出
    pre = out.argmax(dim=1)
    correct = pre.eq(label_test).sum().float().item()
    total_cor += correct
    total_num = index

accuracy = total_cor / len(test_loader.dataset)
print(accuracy)

x, y = next(iter(test_loader))
out = net(x.view(x.shape[0], 28 * 28))
pred = out.argmax(dim=1)
plot_image(x, pred, 'test')
