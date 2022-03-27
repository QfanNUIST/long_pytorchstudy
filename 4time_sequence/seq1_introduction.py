"""
sequence representation：[seq_len,feature_len]
seq_len:表示feature 有多少个
feature_len：表示用多少维的向量表示feature，猫 [1,0] 狗 [0,1]

100天的房价 [100,1],有100个feature，每个feature使用一维向量表示

语言表示：
[words,word vec]
要求：1、尽量降低word vec
2、能够表现语言的语义相近
两种编码方式：
word2vec，glove

加入batch
[word num , b , word vec] : 100月，三个城市，房价
[b , word num , word vec] : 3个城市，100个月，房价
"""

import torch
import torch.nn as nn

"""
为‘hello’，‘word’编码
"""
word_to_index = {"hello": 0, "word": 1}

lookup_tensor = torch.tensor([word_to_index["hello"]], dtype=torch.long) #源码

embeds = nn.Embedding(2, 5)  # 编写两个words，每个word用5维向量表示[seq_len,feature_len] = [2,5]
hello_embed = embeds(lookup_tensor)#编码
print(hello_embed)

"""
使用Glove()编程
"""
from torchnlp.word_to_vector import Glove

vectors = Glove()

vectors['hello']
