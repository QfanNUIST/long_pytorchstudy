'''
将validation训练集上表现最好的模型参数，
运用到validation分割出来的test集上，
但是存在一些问题，由于是random.split所以两次分割的集合可能有些许差别
'''

import torch
import torch.nn as nn
import visdom
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from utils import plot_RGBimage
from resnet18 import Res18
from torchvision.models import resnet18


test_transform = transforms.Compose([
    # 只有resize和规范化
    transforms.Resize([224, 224]),
    transforms.ToTensor(),

    # transforms.Normalize(mean=[0.492, 0.461, 0.417], std=[0.256, 0.248, 0.251])
    # 放在CNN模型中使用Batch_Norm
])
hymenoptera_test = datasets.ImageFolder(root=r"F:\CNN\CNN_classification\ml2021spring-hw3\food-11\validation",
                                        transform=test_transform)

'''将validation数据(660张)分割成，validation（330）和test（330）'''
# valid_divid, test_divid = torch.utils.data.random_split(hymenoptera_test, [330, 330])

test_data = torch.utils.data.DataLoader(hymenoptera_test, batch_size=8, shuffle=False)

'''--------训练模型----------'''
vz = visdom.Visdom()
cnn = Res18()

'''--------test函数----------'''
loss = nn.CrossEntropyLoss()

m_state_dict = torch.load('final_model.pth') #下载存储的模型参数

new_cnn = cnn  #给模型一个新的名字，或者说载入模型，cnn后面不能加括号（）
new_cnn.load_state_dict(m_state_dict) #模型载入参数

new_cnn.eval()

test_loss = []
test_acc = []
for image_test,label_test in test_data:

    pre_test = new_cnn(image_test)
    acc = (pre_test.argmax(dim=-1) == label_test).float().mean()

    loss1_test = loss(pre_test, label_test)

    test_loss.append(loss1_test.item())
    test_acc.append(acc)

epoch_loss_test = sum(test_loss) / len(test_loss)
epoch_acc_test = sum(test_acc) / len(test_acc)

print("test_loss: {:.4f},test_acc: {:.4f}".format( epoch_loss_test, epoch_acc_test))


'''随机寻找一个batch，查看预测的准确性'''
x, y = next(iter(test_data))
out = new_cnn(x)
pred = out.argmax(dim=-1)
plot_RGBimage(x, pred, 'test',y,'true')




# import torch
# from resnet18 import Res18
# from torchvision import  datasets
# from torchvision import transforms
# from torch.utils.data import DataLoader
# from utils import plot_RGBimage
#
# test_transform = transforms.Compose([
#     # 只有resize和规范化
#     transforms.Resize([128, 128]),
#     transforms.ToTensor(),
#
#     # transforms.Normalize(mean=[0.492, 0.461, 0.417], std=[0.256, 0.248, 0.251])
#     # 放在CNN模型中使用Batch_Norm
# ])
# '''--------test函数----------'''
# hymenoptera_test = datasets.ImageFolder(root=r"F:\CNN\CNN_classification\ml2021spring-hw3\food-11\validation",
#                                         transform=test_transform)
# test_data = DataLoader(hymenoptera_test,batch_size=8,shuffle= True)
#
# loss = torch.nn.CrossEntropyLoss()
# cnn = Res18()
# m_state_dict = torch.load('model-3-classify-validation_acc_max.pth')  # 下载存储的模型参数
# new_cnn = cnn  # 给模型一个新的名字，或者说载入模型，cnn后面不能加括号（）
# new_cnn.load_state_dict(m_state_dict)  # 模型载入参数
# new_cnn.eval()
# test_loss = []
# test_acc = []
# for image_test, label_test in test_data:
#     pre_test = new_cnn(image_test)
#     acc = (pre_test.argmax(dim=-1) == label_test).float().mean()
#
#     loss1_test = loss(pre_test, label_test)
#
#     test_loss.append(loss1_test.item())
#     test_acc.append(acc)
#
# epoch_loss_test = sum(test_loss) / len(test_loss)
# epoch_acc_test = sum(test_acc) / len(test_acc)
#
# print("test_loss: {:.4f},test_acc: {:.4f}".format(epoch_loss_test, epoch_acc_test))
#
# x, y = next(iter(test_data))
# out = new_cnn(x)
# pred = out.argmax(dim=-1)
# plot_RGBimage(x, pred, 'test', y, 'true')