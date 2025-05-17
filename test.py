import torch
from torch_geometric.data import Data

print(f"[成功] PyTorch版本: {torch.__version__}")
print(f"[加速] MPS可用: {torch.backends.mps.is_available()}")

# 创建示例图数据
edge_index = torch.tensor([[0, 1], [1, 2]], device='mps')  # 直接放到MPS设备
x = torch.tensor([[1], [2], [3]], device='mps', dtype=torch.float)

data = Data(x=x, edge_index=edge_index.t().contiguous())
print(f"[数据] 示例图数据:\n{data}")

