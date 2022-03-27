'''
where(condition,A,B)满足condition用A中的元素，否则用B中的元素
相比较放入for循环迭代，where高度并行运算速度快
gather(dic,dim,index),在dim中根据index中的元素，索引dic中的元素
'''
import torch
'''where'''
a = torch.randperm(40).view(4,10).float() #4batch,10类
b,ind = a.view(1,40).kthvalue(36)
condition = a>b

torch.where(condition,a,torch.zeros_like(a)) #找出比第36个元素大的所有元素，其余置0

'''torch.gather'''
a = torch.rand(4,26) #26个字母识别
dic = torch.arange(64,64+26).float() #字母的ascll码

pred_value,pred_index = a.topk(4,dim=1) #取输出最可能的四个字母

torch.gather(dic.expand(4,26),dim=1,index = pred_index.long())