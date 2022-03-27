'''
 1、转置 a.t()只能直接在2Dim的tensor上运行
 2、transpose(dim1,dim2),将dim1与dim2的维度互相交换，可以理解为3Dim以上层面的转置
 3、permute(0,2,3,1) 将任意的维度互换，但是如果报错可以加入contiguous()使其连续
'''
import torch
a = torch.rand(4,3,28,28)
print(a.shape)

b = a.transpose(1,3)
print(b.shape)

'''将b再变为a'''
a1 = b.view(4,3*28*28).view(4,3,28,28)
#报错view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
a1 = b.contiguous().view(4,3*28*28).view(4,3,28,28) #加入contiguous()使其连续
#这样恢复的a只是将3*28*28的数据以某种方式重新排列成（4，3，28，28）的维度，但是不一定按照transpose的逆处理过程转置
a2 = b.transpose(1,3)

print(torch.all(torch.eq(a,a1))) #false两个tensor不是完全相等的
print(torch.all(torch.eq(a,a2)))#true两个函数是完全相等的

'''将一张图片从（N,C,H,W）变为（N,H,W,C）(numpy储存图片一般使用这种格式)的tensor
transpose需要两步，而permute只要一步
'''
a = torch.rand(4,3,28,14)

b = a.transpose(1,3) #(N,W,H,C)
b = b.transpose(1,2) #(N,H,W,C)
print(b.shape)
c = a.permute(0,2,3,1) #(N,H,W,C)
