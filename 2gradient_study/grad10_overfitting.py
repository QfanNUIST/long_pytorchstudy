'''过拟合欠拟合
under-fitting ： estimated results < ground truth,train_acc bad, test_acc bad
over-fittting : train good,test_acc bad
train_val_test ----> over-fitting or not

检测over-fitting，交叉验证，train_valadation_test
'''

from torch.utils.data import Dataset, DataLoader
import torchvision
import torch

'''
将train_db,testdb,在train_db上操作
将其变成，train_db1,val_db,train_db

'''
print('train:',len(train_db),'test:',len(test_db))
train_db,val_db = torch.utils.data.random_split(train_db,[5000,1000])
print('db1:',len(train_db),'db2:',len(val_db))
train_loader = torch.utils.data.DataLoader(
    train_db,
    batch_size=batch_size,shuffle=True
)
train_loader = torch.utils.data.DataLoader(
    val_db,
    batch_size=batch_size,shuffle=True
)