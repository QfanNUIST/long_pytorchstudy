'''
范数：norm(n,dim)在dim维度进行n范数，注意dim的理解
mean,sum,min,max,prod,不指定维度会将输入矩阵拉直成一维然后计算全局元素
a.argmax,a.argmin (最大最小值下标索引),同样不指定下表时会将输入矩阵拉直成一维然后计算全局元素

pred_max1 = pred.argmax(dim = 1,keepdim = True)
pred_max1.shape #torch.Size([4, 1]),保持了二维特征

topk(n,dim),在dim维度找出n个最大的数
kthvalue(n,dim),在dim维度找出，按由小到大排序的第n个元素

比较运算torch.all(torch.eq(a,b))
'''

import torch
'''norm(n,dim)在dim维度进行n范数'''
a = torch.ones(8)
b = a.view(2,4)
c = a.view(2,2,2)

a.norm(2)
b.norm(2),b.norm(2,dim = 1) #[2,4]即在4列的维度做2范数，即对列这个维度的所有元素做2范数
c.norm(2),c.norm(2,dim=0)

'''mean,sum,min,max,prod,不指定维度会将输入矩阵拉直成一维然后计算全局元素'''
a = torch.arange(1.0,9).view(2,4).float()
a.mean() #tensor(4.5000)
a.mean(dim = 1) #tensor([2.5000, 6.5000])

a.prod() #内积

'''a.argmax,a.argmin (最大最小值下标索引),同样不指定下表时会将输入矩阵拉直成一维然后计算全局元素'''
a.argmax()
a.argmin()
a.argmax(dim=1)

'''dim和keepdim的区别，minst数据识别为列'''
pred = torch.rand(4,10) #一个batch有四张训练图片，经过网络计算有对应的手写字的概率结果
pred_max = pred.argmax(dim = 1)
pred_max.shape # torch.Size([4])，一维tensor，在后续运算中不方便使用，要将它升维

pred_max1 = pred.argmax(dim = 1,keepdim = True)
pred_max1.shape #torch.Size([4, 1]),保持了二维特征

'''topk(n,dim),在dim维度找出n个最大的数
kthvalue(n,dim),在dim维度找出，按由小到大排序的第n个元素
'''
a = torch.randperm(40).view(4,10) #4batch,10类
[value,index] = a.topk(3,dim=1)  #返回最大的几个值以及对应的索引下表

[value,index] = a.kthvalue(10,dim=1,keepdim=True) #将dim=1的所有元素从小到大排列取第十个元素即最大的元素，取出它的value和index

'''比较运算'''
a = torch.randperm(40).view(4,10) #4batch,10类
a > 0
b = torch.rand(4,10)
torch.eq(a,b)
torch.all(torch.eq(a,b))