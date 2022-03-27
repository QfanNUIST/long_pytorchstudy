'''多输出感知机,多层神经网络'''

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def fun1(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

'''可视化目标函数'''
x = np.arange(-6,6,0.1)
y = np.arange(-6,6,0.1)
print('x,y range',x.shape,y.shape)

X,Y = np.meshgrid(x,y)
print('X,Y.maps',X.shape,Y.shape)
Z = fun1([X,Y])

fig = plt.figure('fun1')
ax = fig.gca(projection = '3d')
ax.plot_surface(X,Y,Z)
ax.view_init(60,-30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

'''使用梯度下降法找出全局最小值'''
x = torch.tensor([0.0,0.0],requires_grad=True)
optimizer = torch.optim.Adam([x],lr=1e-3)
for step in range(1,20000):
    optimizer.zero_grad()
    pred = fun1(x)

    pred.backward()
    optimizer.step()

    if step%2000 ==0:
        a = x[0].item()
        b = x[1].item()
        print('step{}:x={},f(x)={:.2f}'.format(step,[a,b],pred.item()))

