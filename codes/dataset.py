import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset


from typing import List

from utils import read_data, data_split,  generate_samples

import numpy as np
import pandas as pd

class LakeDataset(Dataset):
    def __init__(self, dic):
        self.dic = dic
        
    def __getitem__(self, index):
        tmp_dic = {}
        for key, val in self.dic.items():
            tmp_dic[key] = val[index, :]
        
        return tmp_dic
    
    def __len__(self):
        return self.dic['x'].shape[0]
    
    
def mask_helper(label, mask, mask_rate = 0.99):
    if mask_rate != 0:
        depth, days = label.shape
        label = label.reshape(-1, 1)
        mask = mask.reshape(-1, 1)
        idx = np.random.choice(np.arange(label.shape[0]), 
                               replace=False, 
                               size=int(label.shape[0] * mask_rate))
        label[idx, ] = 0
        mask[idx, ] = 0
        label = label.reshape(depth, days)
        mask = label.reshape(depth, days)
    else:
        pass
    return label, mask

def get_dataloader(path, window_size, strides, batch_size, mask_rate = 0.99, simulate = True):
    full_dict = read_data(path, simulate = simulate)
    train_dict, valid_dict, train_full_dict, test_dict = data_split(full_dict)
    
    train_dict['label'], train_dict['mask'] = mask_helper(train_dict['label'], train_dict['mask'], mask_rate)
    valid_dict['label'], valid_dict['mask'] = mask_helper(valid_dict['label'], valid_dict['mask'], mask_rate)
    train_full_dict['label'], train_full_dict['mask'] = mask_helper(train_full_dict['label'], 
                                                                    train_full_dict['mask'], 
                                                                    mask_rate)
    
    train_samples = generate_samples(train_dict, window_size = window_size, strides = strides)
    valid_samples = generate_samples(valid_dict, window_size = window_size, strides = strides)
    test_samples = generate_samples(test_dict, window_size = window_size, strides = strides)
    


    train_dataset = LakeDataset(train_samples)
    valid_dataset = LakeDataset(valid_samples)
    test_dataset = LakeDataset(test_samples)
    tr_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True)
    va_loader = DataLoader(dataset = valid_dataset, batch_size = batch_size)

    te_loader = DataLoader(dataset = test_dataset, batch_size = batch_size)


    full_samples = generate_samples(train_full_dict, window_size = window_size, strides = strides)
    full_dataset = LakeDataset(full_samples)
    full_loader = DataLoader(dataset = train_dataset, batch_size = batch_size)
    
    return tr_loader, va_loader, full_loader, te_loader
