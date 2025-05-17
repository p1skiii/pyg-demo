from toy_feature_store import ToyFeatureStore
import torch
import pandas as pd


fs = ToyFeatureStore()

# 存入节点0的'x'特征
fs.put_tensor(torch.tensor([1.0, 2.0]), 'paper', 'x', 0)
# 存入节点1的'x'特征
fs.put_tensor(torch.tensor([3.0, 4.0]), 'paper', 'x', 1)
# 存入节点0的'y'特征
fs.put_tensor(torch.tensor([1]), 'paper', 'y', 0)

# 取出节点0的'x'特征
x0 = fs.get_tensor('paper', 'x', 0)
print("节点0的x特征：", x0)
x0 = fs['paper', 'x', 0]
print("节点0的x特征：", x0)

# 取出节点1的'x'特征
x1 = fs.get_tensor('paper', 'x', 1)
print("节点1的x特征：", x1)
x1 = fs['paper', 'x', 1]
print("节点1的x特征：", x1)

# 取出节点0的'y'特征
y0 = fs.get_tensor('paper', 'y', 0)
print("节点0的y特征：", y0)

