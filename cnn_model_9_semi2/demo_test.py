"""
实现semi-supervised
"""
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.utils.data import Dataset
from torchvision import transforms, datasets

from resnet18 import Res18


class PseudoDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, id):
        return self.x[id][0], self.y[id]


def get_pseudo_labels(dataset, model, threshold=0.95):
    # This functions generates pseudo-labels of a dataset using given model.
    # It returns an instance of DatasetFolder containing images whose prediction confidences exceed a given threshold.
    # You are NOT allowed to use any models trained on external data for pseudo-labeling.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Construct a data loader.
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    idx = []
    labels = []

    # Make sure the model is in eval mode.
    model.eval()
    # Define softmax function.
    softmax = nn.Softmax(dim=-1)

    # Iterate over the dataset by batches.
    for i, batch in enumerate(data_loader):  # tqdm 用来显示进度条的，读代码时可以省略
        img, _ = batch

        # Forward the data
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(img.to(device))

        # Obtain the probability distributions by applying softmax on logits.
        probs = softmax(logits)

        # ---------- TODO ----------
        # Filter the data and construct a new dataset.
        for j, x in enumerate(probs):
            if torch.max(x) > threshold:
                idx.append(i * 32 + j)
                labels.append(int(torch.argmax(x)))

    print("\nNew data: {:5d}\n".format(len(idx)))
    # # Turn off the eval mode.
    model.train()
    dataset = PseudoDataset(Subset(dataset, idx), labels)
    return dataset


# def main():
#     test_transform = transforms.Compose([
#         # 只有resize和规范化
#         transforms.Resize([224, 224]),
#         transforms.ToTensor(),
#
#         # transforms.Normalize(mean=[0.492, 0.461, 0.417], std=[0.256, 0.248, 0.251])
#         # 放在CNN模型中使用Batch_Norm
#     ])
#     unlabeled = datasets.ImageFolder(root=r"F:\CNN\CNN_classification\ml2021spring-hw3\food-11\training\unlabeled",
#                                      transform=test_transform)
#     train_dataset = datasets.ImageFolder(root=r"F:\CNN\CNN_classification\ml2021spring-hw3\food-11\training\labeled",
#                                          transform=test_transform)
#     model = Res18()
#     m_state_dict = torch.load('model-3-classify-validation_acc_max.pth')  # 下载存储的模型参数
#     model.load_state_dict(m_state_dict)  # 模型载入参数
#     print("原始训练数据", len(train_dataset))
#     subset = get_pseudo_labels(train_dataset, model)
#     print("增加的pesuo-labeled", len(subset))
#     labeled = ConcatDataset([train_dataset,subset])
#     print("凭借之后",len(labeled))
#
#
#
# main()
