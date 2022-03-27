"""
ht = fw(ht-1,xt)
ht = tanh(Whh*h(t-1) + Wxh*xt)
yt = Why * ht

x = [seq len, batch, feature len]
10个单词，3句话，一个单词用100维向量表示
xt = [3,100]
三句话的100个向量

ht+1 =  xt@wxh + ht@whh ===> [batch,feature len]@[hidden len,feature len].T + [batch,hidden len]@[hidden len, hidden len]


h0 = torch.zeros[batch,hidden len]
out = [seq len , batch , hidden len]
"""

import torch
import torch.nn as nn

rnn = torch.nn.RNN(100,10) #feature len = 100,hidden len = 10
"""
"""
feature_len = 100
hidden_size=20
num_layers=1

rnn = nn.RNN(input_size=feature_len,hidden_size=hidden_size,num_layers=num_layers)

print(rnn)

x = torch.rand(10,3,feature_len)
h0 = torch.zeros(num_layers,3,hidden_size)
out,h = rnn(x,h0)
print(out.shape,h.shape)

"""
nn.RNNCell(100,20)
还是x = [10,3,100]
xt = [3,100]

"""
cell1 = nn.RNNCell(100,20)
h1 = torch.zeros(3,20)

for xt in x:
    h1 = cell1(xt,h1)
print(h1.shape)

"""
两层的RNN网络
"""
cell1 = nn.RNNCell(100,30)
cell2 = nn.RNNCell(30,20)
h1 = torch.zeros(3,30)
h2 = torch.zeros(3,20)

for xt in x:
    h1 = cell1(xt,h1)
    h2 = cell2(h1,h2)
print(h2.shape)
