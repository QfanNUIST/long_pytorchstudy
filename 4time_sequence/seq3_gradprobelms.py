"""
梯度爆炸或者梯度离散
torch.nn.utils.clip_grad_norm(model.parameters(),10)
"""
import torch
loss = criterion(output,y)
model.zero_grad()

loss.backeward()
torch.nn.utils.clip_grad_norm(model.parameters(),10)
optimizer.step()
