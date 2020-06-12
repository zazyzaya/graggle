import os 
import numpy as np

class SparseCSR:
    def __init__(self, row, cols, data):
        self.row = np.load(row, mmap_mode='r')
        self.cols = np.load(cols, mmap_mode='r')
        self.data = np.load(data, mmap_mode='r')
        
    def __getitem__(self, key):
        start = self.row[key]
        end = self.row[key+1]
        
        return DataObject(self.data[start:end], self.cols[start:end])

class DataObject:
    def __init__(self, data, cols):
        self.data = data
        self.indices = cols
        