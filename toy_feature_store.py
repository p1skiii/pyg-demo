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
        self._store[key][index] = tensor
        return self
    
    def get_tensor(self, group_name, after_name, index):
        key = (group_name, after_name)
        return self._store[key][index]

    def __getitem__(self, key):
        group_name, after_name, index = key
        return self.get_tensor(group_name, after_name, index)
        

    