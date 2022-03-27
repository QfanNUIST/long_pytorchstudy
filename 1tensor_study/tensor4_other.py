'''
view = reshape注意在三维以上时需要手动跟踪各个维度的信息 a[4,3,28,28]==>a[4,3*28*28]==>(数据错误)a[4,28,28,3]需要使用transpose
squeeze(dim) :挤压维度
unsqueeze(dim)：在dim处增加一个维度,后面的维度到后面去

'''
import torch

a = torch.rand(4, 1, 28, 28)  # 假设a为minst数据集的一个batch

b = a.view(4 * 1, 28, 28)  # 表示不关注图片的通道，反正只有灰度通道
b = a.view(4 * 1, 28 * 28)  # 表示将其转化为4个28*28大小的向量tensor方便神经网络的全连接

'''squeeze(dim) :挤压维度
unsqueeze(dim)：在dim处增加一个维度
'''
scaler = torch.tensor(1)  # dim = 0
c = scaler.unsqueeze(0).shape  # dim = 1
scaler = torch.tensor([1])  # dim = 1

'''注意unsqueeze(dim)中dim维度的选择'''
c = torch.tensor([2, 3])  # dim =1
d = c.unsqueeze(0)  # (1,2)一行两列的tensor
e = c.unsqueeze(1)  # (2,1)2行一列的tensor

'''实列分析，卷积神经网络中fun函数＋bias'''
a = torch.rand(4, 32, 28, 28)  # 经过卷积运算之后通道数被增加到32
b = torch.rand(32)  # bias为每一个通道一个bias
'''但是一维的b不能和4维的a-tensor做运算，需要给b增加维度'''
b_upscale = b.unsqueeze(1).unsqueeze(2).unsqueeze(0)
b_upscale1 = b.unsqueeze(0).unsqueeze(2).unsqueeze(3)

a_process = a + b_upscale  # b_upscale会自动广播到a的shape （4，32，28，28）

'''b_upscale的dimention是（1，32，1，1）
使用squeeze为b_upscale降维
'''
b_downscale = b_upscale1.squeeze()  # 如果squeeze不指定压缩的维度，就将能压缩的维度都压缩
print(b_downscale.shape)
b_downscale = b_upscale.squeeze(0)

'''扩展方式二 expand,repeat
expend(dim_num1,....,dim_numn)不占用内存，相当于广播
repeat(num1,num2,...,numn)占用内存，参数表示对应的维度重复多少次
'''
a = torch.rand(4, 32, 28, 28)

b = torch.rand(1, 32, 1, 1)
b.expand(4, 32, 28, 28).shape  # 将b展为（4，32，28，28）的形状

c = b.repeat(4, 32, 28, 28)  # 将b展为（4，32*32，28，28）的形状
