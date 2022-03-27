"""
GPU加速
需要送入GPU的参数有，model，lossfunction，data
"""
import torch
import torch.nn as nn

"""模型搭建"""


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)


device = torch.device('cuda:0')  # 数字表示GPU编号
net = MLP().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
criteon = nn.CrossEntropyLoss().to(device)

for epoch in range(epoches):

    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.to(device)
        pass
