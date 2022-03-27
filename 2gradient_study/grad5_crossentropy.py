'''交叉熵损失
熵：-sum(p(log(p)))，衡量不确定性，值越大越稳定
交叉熵cross entropy ：H(p,q) = H(p) + DKL(p|q)
H(p) = -sum(p(log(p))) ，在one hot 矩阵中间，H(p) = 0
DKL(p|q) p,q两个分布之间的KL散度
所以对于分类问题，H(p) = 0，cross entropy 用来衡量pred和label之间分布的重叠度，

eg： 动物五分类
label = [1 0 0 0 0]
pred = [0.4 0.3 0.1 0.1 0.1]
H(p,q) = H(label) + DKL(p|q) = DKL(p|q) =-(1log(0.4) + 0log(0.3) +0log(0.1) + 0log(0.1) + 0log(0.1))
                             = -log(0.4) = 0.916
pred = [0.98 0.01 0 0 0.0.]
H(p,q) = -log(0.98) = 0.02
说明两个分布pred和label 的散度变得越来越小

classification why not use MSE
1、相较cross entropy sigmoid+MSE 容易出现梯度爆炸或者梯度离散
2、cross entropy 梯度更大，更新快，训练时间短
3、但是实际操作时可以先使用MSE做出来
'''

'''
F.cross_entropy（pred） 计算时注意
pred = x@w.t()
不用softmax，因为F.cross_entropy会自动将其softmax，否则就在网络中使用了两次softmax，会出现逻辑问题
'''
