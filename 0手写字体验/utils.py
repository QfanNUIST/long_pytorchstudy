import torch
import matplotlib.pyplot as plt

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

def plot_image(img,label,name):
    fig = plt.figure()
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.tight_layout()
        plt.imshow(img[i][0]*0.3081+0.1307,cmap='gray')
        plt.title("{}:{}".format(name,label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def one_hot(label,depth = 10):
    out = torch.zeros(label.size(0),depth)
    idx = torch.LongTensor(label).view(-1,1)
    out.scatter_(dim=1,index = idx,value = 1)
    return out