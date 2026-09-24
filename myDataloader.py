"""
Return torch dataset Given file path to a cell by PPI gene probability dataframe
"""


import numpy as np
import pandas as pd
import torch as tch
import torch.utils.data as tchud
import sklearn.model_selection as skms
import sklearn.preprocessing as skpre
import myDatasplit as mysplit

class NumpyDataset(tchud.Dataset):
    """
    Return torch dataset, Given X and y numpy array
    """
    def __init__(self, X_arr, y_arr):
        self.X = X_arr
        self.y = y_arr
        self.y = self.y.reshape((len(self.y),1))
     
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return len(self.X)

import h5py

class H5Dataset(tchud.Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        # Open a read-only handle to the h5 file
        self.file = h5py.File(self.h5_path, 'r')
        self.length = self.file['labels'].shape[0]

    def __getitem__(self, index):
        # Lazy read from disk - only loads the specific row into RAM
        x_drug = self.file['drug_features'][index]
        x_cell = self.file['cell_features'][index]
        y = self.file['labels'][index]
        
        return (tch.from_numpy(x_drug).float(), 
                tch.from_numpy(x_cell).float(), 
                tch.tensor([y]).float())
    
    def __len__(self):
        return self.length
    
    def close(self):
        self.file.close()
