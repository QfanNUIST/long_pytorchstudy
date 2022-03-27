'''
将validation训练集上表现最好的模型参数，
运用到validation分割出来的test集上，
但是存在一些问题，由于是random.split所以两次分割的集合可能有些许差别
'''

import torch
import torch.nn as nn
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
hymenoptera_test = datasets.ImageFolder(root=r"F:\CNN\food-11\testing",
                                        transform=test_transform)


'''将validation数据(660张)分割成，validation（330）和test（330）'''
# valid_divid, test_divid = torch.utils.data.random_split(hymenoptera_test, [330, 330])

test_data = torch.utils.data.DataLoader(hymenoptera_test, batch_size=8, shuffle=False)

'''--------训练模型----------'''

cnn = Res18()

'''--------test函数----------'''
loss = nn.CrossEntropyLoss()

m_state_dict = torch.load('model-3-classify-validation_acc_max.pth') #下载存储的模型参数

new_cnn = cnn  #给模型一个新的名字，或者说载入模型，cnn后面不能加括号（）
new_cnn.load_state_dict(m_state_dict) #模型载入参数

new_cnn.eval()

predictions = []
for image_test,label_test in test_data:

    pre_test = new_cnn(image_test)
    with torch.no_grad():
        logits = new_cnn(image_test)

    # Take the class with greatest logit as prediction and record it.
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
# Save predictions into the file.
with open("pre/predict.csv", "w") as f:

    # The first row must be "Id, Category"
    f.write("Id,Category\n")

    # For the rest of the rows, each image id corresponds to a predicted class.
    for i, pred in  enumerate(predictions):
         f.write(f"{i},{pred}\n")





