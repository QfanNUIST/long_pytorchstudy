"""
LSTM解析
在RNN的基础上给下一层的每一个输入加入一个控制门

门 ： sigmoid（）
ft = sigmoid（Wf*[ht-1,xt] + bf）
Ct-1_head = Ct-1 * ft

input gate and cell gate
input gate:
it = sig(Wi * [ht-1,xt] + bi)

cell gate
Ct_head = tanh(Wc *[ht-1,xt] + bc )

C_head = Ct_head * it

Ct = C_head + Ct-1_head

output: ht
Ot = sig(Wo * [ht-1,xt] + bo)
ht = Ot * tanh(Ct)

"""

import torch
from torch import nn

"""
nn.LSTM类，输入的参数和RNN相同
nn.LSTM(feature_len, hidden_len, num_layers)
初始化时输入的参数

LSTM.forward()
运行时输入的参数
out,(ht,ct) = lstm(x,[ht_1,ct_1])
x:[seq,b,vec]
h/c: [num_layer,b,h]
out:[seq,b,h]
"""
lstm = nn.LSTM(input_size=100,hidden_size=20,num_layers=4)
print(lstm)
x = torch.rand(10,3,100)
out,(h,c) = lstm(x)
print(out.shape, h.shape, c.shape)


"""
nn.LSTMCell:
初始化时是一样的nn.LSTMCell(input_size, hidden_size, num_layers)

nn.forward():
ht,ct = lstmcell(xt, [ht_1,ct_1])

x = [10,3,100]
xt = [3,100]
ht_1,ct_1 = [3,20]
"""

print('one layer lstm')
x = torch.rand(10,3,100)
cell = nn.LSTMCell(100,20,1)
h = torch.rand(3,20)
c = torch.rand(3,20)

for xt in x :
    h,c = cell(xt,[h,c])
print(h.shape,c.shape)

print('two layers lstm')

x = torch.rand(10,3,100)
cell1 = nn.LSTMCell(100,30,1)
cell2 = nn.LSTMCell(30,20,1)
h1 = torch.rand(3,30)
c1 = torch.rand(3,30)

h2 = torch.rand(3,20)
c2 = torch.rand(3,20)

for xt in x :
    h1,c1 = cell1(xt,[h1,c1])
    h2,c2 = cell2(h1,[h2,c2])
print(h2.shape,c2.shape)