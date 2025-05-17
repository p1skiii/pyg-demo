import torch
import numpy as np
import pandas as pd

class ToyGraphStore:
    def __init__(self):
        # 存储不同类型的边
        self._store = {}
    
    def put_edge_index(self, edge_index, edge_type):
        if not isinstance(edge_index, torch.Tensor):
            raise TypeError("edge_index must be a torch.Tensor")
        if not isinstance(edge_type, tuple) or len(edge_type) != 3:
            raise ValueError("edge_type must be a tuple of (source_node_type, relation_type, target_node_type)")
        
        self._store[edge_type] = edge_index
        return self
    