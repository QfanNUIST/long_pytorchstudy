'''
这一章节主要学习tensor中的拼接与拆分
拼接 cat()在dim出拼接在一起（RGB）,stack()在dim处unsqueeze一个维度在拼接（Nbatch）
拆分 split()在dim处拆分后每个batch有n个图片,chunk在dim处将所有数据分成n个batch
'''
import torch
'''拼接操作1  cat([tensor1,tensor2],dim)'''
a = torch.rand(4,32,8) #1-4班的成绩
b = torch.rand(3,32,8) #5-7班的成绩
scor = torch.cat([a,b],dim=0) #在dim = 0,班级维度进行拼接[7,32,8]
print(scor.shape)

'''拼接2  stack([tensor1,tensor2],dim)
在dim处创建新维度将两个tensor链接起来
'''
a = torch.rand(3,28,28)#一张图片
b = torch.rand(3,28,28)
batch = torch.stack([a,b],dim=0) #将两张图合为一个batch [2,3,28,28]

'''拆分1 split([len1,len2,...,len_n],dim)在dim维度将原来的矩阵拆分成，len1，、、、，len_n'''
train_data = torch.rand(8,3,14,14) #假设训练数据一共有8张RGB图像
[batch1,batch2,batch3,batch4] = train_data.split([1,2,3,2],dim = 0) #将训练数据分为四个batch实际意义，文科，理科，教改班
[batch1,batch2,batch3,batch4] = train_data.split(2,dim=0) #将数据在dim = 0处以长度为2分为4个batch

'''按数量拆分chunk'''
aa,bb = train_data.chunk(2,dim = 0) #将epoch分为两个batch，长度没有限制


