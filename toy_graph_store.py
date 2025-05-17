import torch
import numpy as np
import pandas as pd

class ToyFeatureStore:
    def __init__(self):
        self._store = {}
        # 键是边类型 (source_node_type, relation_type, target_node_type)
        # 值是边索引 tensor，shape 为 [2, num_edges]

    def put_edge_index(self, edge_index, )