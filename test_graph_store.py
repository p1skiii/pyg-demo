import torch
import numpy as np
import pandas as pd

class ToyGraphStore:
    def __init__(self):
        # 存储不同类型的边
        self._store = {}
    
    def put_edge_index(self, edge_index, edge_type, mode='append'):
        if not isinstance(edge_index, torch.Tensor) or edge_index.shape[0] != 2:
            raise TypeError("edge_index must be a torch.Tensor")
        if not isinstance(edge_type, tuple) or len(edge_type) != 3:
            raise ValueError("edge_type must be a tuple of (source_node_type, relation_type, target_node_type)")
        if mode == 'append' and edge_type in self._store:
            existing_edge_index = self._store[edge_type]        
            self._store[edge_type] = torch.cat([existing_edge_index, edge_index], dim=1)
        else:
            self._store[edge_type] = edge_index
        return self
    
    def get_edge_index(self, edge_type, layout='coo'):
        if edge_type not in self._store:
            return None
        
        if layout.lower() != 'coo':
            raise ValueError(f"当前只支持'coo'格式，不支持'{layout}'格式")
            
        return self._store[edge_type]
    
    def __getitem__(self, key):
        edge_type, layout = key
        return self.get_edge_index(edge_type, layout)
