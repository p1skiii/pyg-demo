import torch
import numpy as np
import pandas as pd

class ToyFeatureStore:
    def __init__(self):
        self._store = {}

    def put_tensor(self, tensor, group_name, after_name, index):
        key = (group_name, after_name)

        if key not in self._store:
            self._store[key] = {}
        if isinstance(index, int): 
            // torch.tensor([1.0, 2.0]), 'paper', 'x', 0
            self._store[key][index] = tensor
        elif isinstance(index, (list, torch.Tensor)): 
            // torch.tensor([[1.0, 2.0], [3.0, 4.0]]), 'paper', 'x', [0, 1])
            // 
            for i, idx in enumerate(index):
                self._store[key][idx] = tensor[i]
        else:   
            raise ValueError("Index must be an int or a list of ints")
        return self
    
    def get_tensor(self, group_name, attr_name, index):
        key = (group_name, attr_name)
        if isinstance(index, int):
            return self._store[key][index]
        elif isinstance(index, (list, torch.Tensor)):
            return torch.stack([self._store[key][int(i)] for i in index])
        elif isinstance(index, slice):
            indices = range(index.start or 0, index.stop, index.step or 1)
            return torch.stack([self._store[key][i] for i in indices])
        elif index is None:
            all_indices = sorted(self._store[key].keys())
            return torch.stack([self._store[key][i] for i in all_indices])
        else:
        raise TypeError("不支持的 index 类型")

    def __getitem__(self, key):
        group_name, after_name, index = key
        return self.get_tensor(group_name, after_name, index)



    
        

    