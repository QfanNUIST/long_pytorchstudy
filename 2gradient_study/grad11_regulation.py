'''regelazation 原理

方法：more data
constraint model complexity :shallow,regularization
Dropout
data argumentation
early stoppomh
'''
import torch


'''weight_decay = 0.01表示L2regularition的权重'''
optomizer = torch.optim.SGD(net.parameters(),lr = learning_rate,weight_decay = 0.01)

'''在loss衰减较慢的地方，降低学习率'''
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate,momentum=momentum,weight_decay=weight_decay)
scheduler = torch.nn.ReduceLROnPlateau(optimizer,'min')
for epoch in range(epoches):
    #train
    pass
    loss = not ...
    scheduler.step(loss) #scheduler监视loss是否很长时间没有下降如果是，就将lr减少

'''stop early'''

'''Dropout,剪枝
在valid和test时，要使用不剪枝的模型
'''
net_dropped = torch.nn.Sequential(
    torch.nn.Linear(784,200),
    torch.nn.Dropout(0.5),#drop 50% of the neuron
    torch.nn.ReLU(),
    torch.nn.Linear(200,200),
    torch.nn.Dropout(0.5),#drop 50% of the neuron
    torch.nn.ReLU(),
    torch.nn.Linear(200,10),
)
'''测试时保留所有链接'''
for epoch in range(epoches):
    #train
    net_dropped.train()
    for batch_index,(data,label) in enumerate(train_loader):
        pass #train
    '''切换剪枝模式，test不剪枝'''
    net_dropped.eval()
    test_loss = 0
    correct = 0
    for data,target in test_loader:
        pass