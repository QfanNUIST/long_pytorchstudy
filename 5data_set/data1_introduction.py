"""
自己编写 Dataset 类
主要自己实现的
_len_
_getitem_
"""
from torch.utils.data import Dataset

"""
return len(self.samples) 确定的迭代次数，同时会返回idx，告诉你现在运行到哪了，但是不能索引
"""


class Numbersdataset(Dataset):
    def __init__(self, training=True):
        if training:
            self.samples = list(range(1, 1001))
        else:
            self.samples = list(range(1001, 1501))

        def _len_(self):
            return len(self.samples)

        def _getitem_(self, idx):
            return self.samples[idx]
