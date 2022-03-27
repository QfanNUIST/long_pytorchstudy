"""
主执行程序
"""
"""
目前存在问题不能输出printl里面的内容
"""
import torch
import visdom
from resnet18 import Res18
from data2_download_data import Getdata, Gettestdata
from torch.utils.data import DataLoader


epoches = 10
lr = 1e-3
weight_decay = 1e-5

"""载入数据"""
root = r"F:\CNN\CNN_classification\ml2021spring-hw3\food-11\training\labeled"
root1 = r"F:\CNN\CNN_classification\ml2021spring-hw3\food-11\validation"
save_root = r"F:\pytorchstudy\5data_set"
train_data_path = Getdata(root, save_root, 224, 'train')
train_data = DataLoader(train_data_path, batch_size=32, shuffle=True)
valid_data_path = Getdata(root, save_root, 224, 'val')
valid_data = DataLoader(valid_data_path, batch_size=32, shuffle=False)
test_data_path = Gettestdata(root1, save_root, 224, 'test')
test_data = DataLoader(test_data_path, batch_size=32, shuffle=False)
"""加载模型"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("{} is used".format(device))
model = Res18().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criteon = torch.nn.CrossEntropyLoss()


# 定义valid和test函数
def evaluate(modelname, loader):
    modelname.eval()
    valid_loss_batch = []
    valid_acc_batch = []
    for x, y in loader:
        x,y = x.to(device),y.to(device)
        logits = modelname(x)
        loss = criteon(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        valid_loss_batch.append(loss.item())
        valid_acc_batch.append(acc)
    valid_loss = sum(valid_loss_batch) / len(valid_loss_batch)
    valid_acc = sum(valid_acc_batch) / len(valid_acc_batch)
    return valid_loss, valid_acc


"""可视化"""
viz = visdom.Visdom()
"""可视化参数"""
'''loss'''
viz.line([[0., 0.]], [0.], win='loss1', opts=dict(title='train&valid loss', legend=['train loss', 'valid loss']))
# [y1,y2]=[0., 0.],[x]=[0.],x在最后
# 这个绘图窗口的名字：loss
# 这个图片的名字 ：train&valid loss
'''acc可视化'''
viz.line([[0., 0.]], [0.], win='acc1', opts=dict(title='train&valid acc', legend=['train acc', 'valid acc']))
# [y1,y2]=[0., 0.],[x]=[0.],x在最后
# 这个绘图窗口的名字：loss
# 这个图片的名字 ：train&valid loss



train_loss_epoch = []
train_acc_epoch = []
valid_loss_epoch = []
valid_acc_epoch = []
valid_acc_max = -1
for epoch in range(epoches):
    train_loss_batch = []
    train_acc_batch = []
    model.train()
    for step, (image, label) in enumerate(train_data):
        logits = model(image)
        loss = criteon(logits, label)
        acc = (logits.argmax(dim=-1) == label).float().mean()
        train_loss_batch.append(loss.item())
        train_acc_batch.append(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 10 == 0:
            print("epoch:[{}] , step:[{}],loss:{:.4f}".format(epoch+1, step+1, sum(train_loss_batch) / len(train_loss_batch)))
    train_loss = sum(train_loss_batch) / len(train_loss_batch)
    train_acc = sum(train_acc_batch) / len(train_acc_batch)
    train_loss_epoch.append(train_loss)
    train_acc_batch.append(train_acc)
    print("train_batch_loss:{:.4f}, train_batch_acc:{:.4f}".format(train_loss, train_acc))

    "-------validation moudel--------"
    valid_loss, valid_acc = evaluate(model,valid_data)
    valid_loss_epoch.append(valid_loss)
    valid_acc_epoch.append(valid_acc)
    print("valid_epoch_loss:{:.4f}, valid_epoch_acc:{:.4f}".format(valid_loss, valid_acc))
    if valid_acc_max < valid_acc:
        valid_acc_max = valid_acc
        print("valid acc max:{}".format(valid_acc_max))
        torch.save(model.state_dict(), 'best_acc.mdl')

    viz.line([[train_loss, valid_loss]], [epoch + 1], win='loss1', update='append')
    # update='append' 在后面连续增加更新
    viz.line([[train_acc, valid_acc]], [epoch + 1], win='acc1', update='append')


"""------test------"""
print('best acc: ', valid_acc_max)

model.load_state_dict(torch.load('best_acc.mdl'))
test_loss, test_acc = evaluate(model, test_data)
print("valid_batch_loss:{:.4f}, valid_batch_acc:{:.4f}".format(test_loss, test_acc))