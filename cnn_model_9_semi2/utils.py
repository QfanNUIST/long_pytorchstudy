import torch
from torch import nn
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

'''
这个文件储存一些常用的函数
plot_curve：绘画data的曲线
plot_image：画图片
one_hot：用scatter_完成对label  onehot的编码
'''

def plot_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)),data,color = 'blue')
    plt.legend(['value'],loc = 'upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()

def plot_subcurve(data1,data2,name1,name2):
    fig = plt.figure()
    plt.plot(range(len(data1)),data1,color = 'blue')
    plt.plot(range(len(data1)), data2, color='red')

    plt.legend(['{}'.format(name1),'{}'.format(name2)],loc = 'upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()

def plot_image(img,label,name):
    fig = plt.figure()
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.tight_layout()
        plt.imshow(img[i][0],cmap='g')
        plt.title("{}:{}".format(name,label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def plot_RGBimage(img,label1,name1,label2,name2):
    img = img.permute(0,2,3,1)
    fig = plt.figure()
    for i in range(8):
        plt.subplot(2,4,i+1)
        plt.tight_layout()
        plt.imshow(img[i])
        plt.title("{}:{},{}:{}".format(name1,label1[i].item(),name2,label2[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def one_hot(label,depth = 10):
    out = torch.zeros(label.size(0),depth)
    idx = torch.LongTensor(label).view(-1,1)
    out.scatter_(dim=1,index = idx,value = 1)
    return out

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)