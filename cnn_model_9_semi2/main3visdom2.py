"""
导入使用监督学习学到的参数
modetransl_before_split-validation_acc_0.7130.pth
使用半监督学习增加训练数据，削弱过拟合现象
"""
'''在cnn_model2的基础上'''

import torch
from torch.autograd.grad_mode import no_grad
import torch.nn as nn
import visdom
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms, datasets

from resnet18 import Res18
from utils import plot_RGBimage
from demo_test import get_pseudo_labels


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"    #使用1号


'''数据处理'''
batch_size = 8

'''------data argumentation------'''
data_dataarguentation = transforms.Compose([
    transforms.Resize([int(224 * 1.25), int(224 * 1.25)]),

    transforms.RandomHorizontalFlip(0.5),
    # # 水平翻转
    transforms.RandomVerticalFlip(0.5),
    # 垂直翻转
    transforms.RandomRotation(15),
    # 在[-15,15]之间随机旋转
    # transforms.RandomRotation([0,90,180,270]),
    # 在[0,90,180,270]之间随机旋转一次
    transforms.RandomCrop([224, 224]),
    # 裁剪部分

    transforms.ToTensor(),

    # transforms.Normalize(mean=[0.492, 0.461, 0.417], std=[0.256, 0.248, 0.251])
    # 放在CNN模型中使用Batch_Norm

])
'''加入valid和test的transformer'''
test_transform = transforms.Compose([
    # 只有resize和规范化
    transforms.Resize([224, 224]),
    transforms.ToTensor(),

    # transforms.Normalize(mean=[0.492, 0.461, 0.417], std=[0.256, 0.248, 0.251])
    # 放在CNN模型中使用Batch_Norm
])

hymenoptera_dataset = datasets.ImageFolder(root=r"food-11/training/labeled",transform=data_dataarguentation)
hymenoptera_test = datasets.ImageFolder(root=r"food-11/validation",transform=test_transform)
hymenoptera_unlabeled = datasets.ImageFolder(root=r"food-11/training/unlabeled",transform=test_transform)

'''将validation数据(660张)分割成，validation（330）和test（330）'''
# train_divid, valid_divid = torch.utils.data.random_split(hymenoptera_dataset, [2400, 680])

# train_data = torch.utils.data.DataLoader(train_divid, batch_size=32, shuffle=True)
valid_data = torch.utils.data.DataLoader(hymenoptera_test, batch_size=batch_size, shuffle=False)
# test_data = torch.utils.data.DataLoader(hymenoptera_test, batch_size=32, shuffle=False)

"""------在这儿可以改变使用的模型 My_CNN() Resnet18 -------"""

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Res18().to(device=device)
model.device = device
m_state_dict = torch.load('model-3-classify-validation_acc_max.pth')  # 下载存储的模型参数
model.load_state_dict(m_state_dict)  # 模型载入参数
cnn = model
'''设置优化其，loss'''
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.0001, weight_decay=3e-5)
# 初始lr = 0.0001
loss = nn.CrossEntropyLoss()

'''测试卷积运算代码'''
epoches = 200 # 代码训练的轮数
'''加入n_batch模型'''

"""添加visdom"""
viz = visdom.Visdom()
'''-----train model------'''
# train_loss_save = []
# valid_loss_save = []
# 方便最后做出损失函数图像
# train_acc_save = []
# valid_acc_save = []
valid_acc_max = -1
# 将概率最大的看作预测出来的类

"""可视化参数"""
'''loss'''
viz.line([[3., 3.]], [0.], win='loss1', opts=dict(title='train&valid loss', legend=['train loss', 'test_loss']))
viz.line([[0., 0.]], [0.], win='acc1', opts=dict(title='train&valid acc', legend=['train acc', 'test acc']))

for epoch in range(epoches):
    step = 1  # 初始化内部的index参数，主要用来监控，已经处理的图片数量
    train_batch_loss = []  # 储存一个epoch中每一个batch所有的损失
    train_batch_acc = []
    if epoch % 20 == 0:
        print("-----loading new data -----")
        subset = get_pseudo_labels(hymenoptera_unlabeled, cnn)
        train_set = ConcatDataset([hymenoptera_dataset, subset])
        print("epoch[{}], train_set length[{}]".format(epoch+1, len(train_set)))
        train_data = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    cnn.train()
    # 对自己定义的cnn模型，切换至训练模式
    for image, label in train_data:
        image = image.to(device)
        label = label.to(device)

        pre = cnn(image)

        # pre = pre.view(4,1)
        loss1 = loss(pre, label)
        # loss_save.append(np.array(loss1.data))

        acc = (pre.argmax(dim=-1) == label).float().mean()
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()
        nn.utils.clip_grad_norm_(cnn.parameters(), max_norm=10)
        # 对梯度进行限制，不超过10，防止梯度爆炸或者梯度消失

        train_batch_loss.append(loss1.item())
        train_batch_acc.append(acc)

        if step % 30 == 0:
            print("epoch:[{}],step:[{}] train_batch_loss: {:.4f}".format(epoch + 1, step,
                                                                         sum(train_batch_loss) / len(train_batch_loss)))
            # 输出每50(epoch)x8(batch) = 400 张图片的损失
        step = step + 1

    '''------------计算上一个epoch的loss，acc------------'''
    train_epoch_loss = sum(train_batch_loss) / len(train_batch_loss)
    train_epoch_acc = sum(train_batch_acc) / len(train_batch_acc)
    # train_acc_save.append(train_epoch_acc)
    # train_loss_save.append(train_epoch_loss)
    # 储存acc和loss方便画图
    print("epoch：[{}] train_epoch_loss: {:.4f} , train_epoch_acc: {:.4f}".format(epoch + 1, train_epoch_loss,
                                                                                 train_epoch_acc))  # 输出每50(epoch)x8(batch) = 400 张图片的损失

    '''-----------validation测试--------------'''
    valid_loss = []
    valid_acc = []

    cnn.eval()
    with torch.no_grad():
        for image_valid, label_valid in valid_data:
            image_valid = image_valid.to(device)
            label_valid = label_valid.to(device)

            pre_valid = cnn(image_valid)
            acc = (pre_valid.argmax(dim=-1) == label_valid).float().mean()
            loss1_valid = loss(pre_valid, label_valid)

            valid_loss.append(loss1_valid.item())
            valid_acc.append(acc)

    valid_epoch_loss = sum(valid_loss) / len(valid_loss)
    valid_epoch_acc = sum(valid_acc) / len(valid_acc)
    # valid_loss_save.append(valid_epoch_loss)
    # valid_acc_save.append(valid_epoch_acc)
    print("epoch:[{}] valid_loss: {:.4f},valid_acc: {:.4f}".format(epoch + 1, valid_epoch_loss, valid_epoch_acc))
    

    """visdom 可视化"""
    '''loss可视化'''
    viz.line([[train_epoch_loss,valid_epoch_loss]], [epoch], win='loss1', update='append')
    viz.line([[train_epoch_acc.cpu(),valid_epoch_acc.cpu()]], [epoch], win='acc1', update='append')

    if valid_epoch_acc > valid_acc_max:
        valid_acc_max = valid_epoch_acc
        print("valid_acc_max:[{}:.4f]".format(valid_acc_max))
        model_name = 'model-3-classify-validation_acc_max1.pth'
        torch.save(cnn.state_dict(), model_name)  # 保存全部模型参数
    else:
        print("valid_acc_final:[{:.4f}]".format(valid_epoch_acc))
        model_name = 'final_model.pth'
        torch.save(cnn.state_dict(), model_name)  # 保存全部模型参数

# # 保存最终的模型
# model_name = 'model-3.3-classify.pth'.format(epoches, round(train_batch_loss))
# torch.save(cnn.state_dict(), model_name)  # 保存全部模型参数



"------test-------"
hymenoptera_test = datasets.ImageFolder(root=r"food-11/testing",transform=test_transform)
test_data = torch.utils.data.DataLoader(hymenoptera_test, batch_size=8, shuffle=False)
'''--------test函数----------'''
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Res18().to(device=device)
model.device = device
m_state_dict = torch.load('model-3-classify-validation_acc_max1.pth') #下载存储的模型参数
model.load_state_dict(m_state_dict)  # 模型载入参数
new_cnn = model  #给模型一个新的名字，或者说载入模型，cnn后面不能加括号（）

new_cnn.eval()
with no_grad():
    predictions = []
    for image_test,label_test in test_data:
        image_test = image_test.to(device)
        label_test = label_test.to(device)

        pre_test = new_cnn(image_test)
        with torch.no_grad():
            logits = new_cnn(image_test)

        # Take the class with greatest logit as prediction and record it.
        predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    # Save predictions into the file.
with open("pre1/predict2.csv", "w") as f:

    # The first row must be "Id, Category"
    f.write("Id,Category\n")

    # For the rest of the rows, each image id corresponds to a predicted class.
    for i, pred in  enumerate(predictions):
        f.write(f"{i},{pred}\n")