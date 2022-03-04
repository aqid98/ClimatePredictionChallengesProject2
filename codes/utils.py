import os, random, warnings
import numpy as np
import pandas as pd
import datetime
import torch
# import cv2
import re
import matplotlib.pyplot as plt
import sciencebasepy
import urllib.request
from sklearn import preprocessing
from urllib.request import urlopen
from zipfile import ZipFile


import time
import inspect
from tqdm import tqdm
import torch
import torch.nn as nn

def get_module_forward_input_names(module: nn.Module):
    params = inspect.signature(module.forward).parameters
    param_names = [k for k, v in params.items() if not str(v).startswith("*")]
    return param_names


def model_inference(model, test_loader, device = 'cpu'):
    epoch_loss = 0
    tic = time.time()
    
    input_names = get_module_forward_input_names(model)
    with tqdm(test_loader, disable = True) as it:
        for batch_no, data_entry in enumerate(it, start = 1):
            inputs = [data_entry[k].to(device) for k in input_names]
            with torch.no_grad():
                output = model(*inputs)
            target = data_entry['label'].to(device)
            weight = data_entry['mask'].to(device)
            loss = torch.sqrt(torch.sum(weight * (output - target) ** 2) / torch.sum(weight))
            epoch_loss += loss.item()

        lv = epoch_loss/len(test_loader)

    toc = time.time()
    return lv, toc - tic

def download_data(data_dir):
    if os.listdir(data_dir) == []: ## data not downloaded yet
        print('Data folder is empty! Download the files now!')
        # set the url
        zipurl = 'https://github.com/leap-stc/LEAPCourse-Climate-Pred-Challenges/raw/main/Project-StarterCodes/Project2-PhysicsML/data/numpy_files.zip'
        # download the file from the URL
        zipresp = urlopen(zipurl)
        # create a new file on the hard drive
        tempzip = open(data_dir + 'numpy_files.zip', "wb")
        # write the contents of the downloaded file into the new file
        tempzip.write(zipresp.read())
        # close the newly-created file
        tempzip.close()
        # re-open the newly-created file with ZipFile()
        zf = ZipFile(data_dir + 'numpy_files.zip')
        # extract its contents into <extraction_path>
        # note that extractall will automatically create the path
        zf.extractall(path = data_dir)
        # close the ZipFile instance
        zf.close()
        print('Files all downloaded!')
        
def read_data(data_dir, simulate = True):
    # load data
    x_full = np.load(data_dir + '/processed_features.npy') #standardized inputs
    x_raw_full = np.load(data_dir + '/features.npy') #raw inputs
    if simulate:
        diag_full = np.load(data_dir + '/diag.npy') 
        label_full = np.load(data_dir + '/labels.npy') #simulated lake temperatures

        # process data
        mask_full = np.ones(label_full.shape) # no missing values to mask for simulated data
        phy_full = np.concatenate((x_raw_full[:,:,:(-2)], diag_full), axis=2) 
        ## phy: 4-air temp, 5-rel hum, 6-wind speed, 9-ice flag
    else:
        diag_full = np.load(data_dir + '/diag.npy')
        label_full = np.load(data_dir + '/Obs_temp.npy') # real observation data
        mask_full = np.load(data_dir + '/Obs_mask.npy') # flags of missing values
        phy_full = np.concatenate((x_raw_full[:,:,:-2], diag_full), axis = 2) #physics variables
    full_dict = {
        'x':x_full,
        'x_raw': x_raw_full,
        'diag': diag_full,
        'label':label_full,
        'mask':mask_full,
        'phy':phy_full
    }
    return full_dict



def data_split(full_dict):
    N = full_dict['x'].shape[1]
    idx_tr, idx_va, idx_te = (int(N/3), int(N/3*2), N)
    
    train_dict = {}
    valid_dict = {}
    test_dict = {}
    train_full_dict = {}
    for key, val in full_dict.items():
        train_dict[key] = val[:, :idx_tr]
        valid_dict[key] = val[:, idx_tr:idx_va]
        train_full_dict[key] = val[:, :idx_va]
        
        test_dict[key] = val[:, idx_va:]
    return train_dict, valid_dict, train_full_dict, test_dict

def generate_samples(dict_, window_size = 352, strides = 352//2,):
    sample_dict = {}
    for key, val in dict_.items():
        sample_dict[key] = []
    
    size = dict_['x'].shape[1]
    for key, val in dict_.items():
        loc = 0
        while loc + window_size < size:
            
            tmp_array = val[:, loc: loc + window_size]
            tmp_array = np.expand_dims(tmp_array, axis = 0)
            sample_dict[key].append(tmp_array)
            
            loc += strides
        sample_dict[key] = np.vstack(sample_dict[key]).astype(np.float32)
    return sample_dict
        
    