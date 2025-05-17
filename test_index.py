import torch

# 测试1：index 是连续的 [0, 1]
tensor1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
index1 = [0, 1]

print("测试1：index = [0, 1], tensor = ")
print(tensor1)

for i, idx in enumerate(index1):
    print(f"i={i}, idx={idx}, tensor[i]={tensor1[i]}")
    # 存储：self._store[key][idx] = tensor[i]
    # 也就是：self._store[key][0] = tensor[0] = [1.0, 2.0]
    #       self._store[key][1] = tensor[1] = [3.0, 4.0]

print("\n" + "="*50 + "\n")

# 测试2：index 不连续 [5, 8]
tensor2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
index2 = [5, 8]

print("测试2：index = [5, 8], tensor = ")
print(tensor2)

for i, idx in enumerate(index2):
    print(f"i={i}, idx={idx}, tensor[i]={tensor2[i]}")
    # 存储：self._store[key][idx] = tensor[i]
    # 也就是：self._store[key][5] = tensor[0] = [5.0, 6.0]
    #       self._store[key][8] = tensor[1] = [7.0, 8.0]


