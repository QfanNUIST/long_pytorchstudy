"""
好评差评实战
数据集：IMDB

"""

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchtext import data

TEXT = data.Field(tokenize = 'spacy')
LABEL = data.LabelField(dtype = torch.float)

