import torch
from toy_feature_store import ToyFeatureStore

# 创建一个 ToyFeatureStore 并存入一些数据
fs = ToyFeatureStore()

# 存入几个节点的特征
fs.put_tensor(torch.tensor([1.0, 2.0]), 'paper', 'x', 0)
fs.put_tensor(torch.tensor([3.0, 4.0]), 'paper', 'x', 1)
fs.put_tensor(torch.tensor([5.0, 6.0]), 'paper', 'x', 2)

print("已存入节点0、1、2的特征")

# 测试1：使用 list 类型的 index 读取
index_list = [0, 2]  # 只读节点0和节点2
print("\n测试1：使用 list 读取")
print(f"index_list = {index_list}")

# 用列表推导式模拟 get_tensor 中的代码
result_list = [fs._store[('paper', 'x')][int(i)] for i in index_list]
print("列表推导式结果（逐个tensor）：")
for t in result_list:
    print(t)

# stack 结果
result_stack_list = torch.stack(result_list)
print("stack结果（合并后的tensor）：")
print(result_stack_list)

# 直接调用 get_tensor
result_get_list = fs.get_tensor('paper', 'x', index_list)
print("get_tensor返回结果：")
print(result_get_list)

# 测试2：使用 torch.Tensor 类型的 index 读取
index_tensor = torch.tensor([0, 2])  # 同样只读节点0和节点2
print("\n测试2：使用 torch.Tensor 读取")
print(f"index_tensor = {index_tensor}")

# 用列表推导式模拟 get_tensor 中的代码
result_tensor = [fs._store[('paper', 'x')][int(i)] for i in index_tensor]
print("列表推导式结果（逐个tensor）：")
for t in result_tensor:
    print(t)

# stack 结果
result_stack_tensor = torch.stack(result_tensor)
print("stack结果（合并后的tensor）：")
print(result_stack_tensor)

# 直接调用 get_tensor
result_get_tensor = fs.get_tensor('paper', 'x', index_tensor)
print("get_tensor返回结果：")
print(result_get_tensor)

# 确认两种方式结果相同
print("\n两种索引方式结果是否相同：", torch.equal(result_stack_list, result_stack_tensor))
