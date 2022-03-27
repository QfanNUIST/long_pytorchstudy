'''
resnet可以平滑loss函数，更有利于找到全局最小值
SGD没办法跳出鞍点

激活函数：
sigmoid 1/(1+exp(-x)) :会出现梯度消失，一般用于分类任务的最后一层
tanh  (exp(x) - exp(-x))/(exp(x) + exp(-x)) == 2sigmoid(2x) - 1 :常用于RNN
RLU  f(x)= 0  x<0
     f(x) = 1 x>=0


'''
import torch

'''sigmoid函数'''
z = torch.linspace(-100,100,10)
grad = torch.sigmoid(z)

'''Relu'''
torch.relu(z)
