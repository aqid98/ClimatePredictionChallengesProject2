import inspect
import logging
import time

from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import pandas as pd

import gc


def get_module_forward_input_names(module: nn.Module):
    params = inspect.signature(module.forward).parameters
    param_names = [k for k, v in params.items() if not str(v).startswith("*")]
    return param_names

logging.basicConfig(level = 'INFO', # DEBUG
        format = "%(asctime)s %(levelname)s:%(lineno)d] %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S")

class Trainer:
    def __init__(self, 
                 epochs = 500,
                 learning_rate = 1e-2,
                 device = 'cpu',
                 early_stopping_patience = 3):
        
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device
        self.early_stopping_patience = early_stopping_patience
    
    def __call__(self, 
                 model,
                 loss_func,
                 train_loader, 
                 valid_loader):
        is_validation_available = valid_loader is not None
        model.to(self.device)
        self.input_names = get_module_forward_input_names(model)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        patience = 0
        
        epoch_info = {
                "epoch_no": -1,
                "loss": np.Inf,}
        
        for epoch_no in range(self.epochs):

            epoch_loss = self.loop(epoch_no, model, loss_func, train_loader, optimizer, is_train = True)
            if is_validation_available:
                epoch_loss = self.loop(
                    epoch_no, model, loss_func, valid_loader, optimizer, is_train=False
                    )
            if epoch_loss < epoch_info['loss']:
                epoch_info['loss'] = epoch_loss
                epoch_info['epoch_no'] = epoch_no
                patience = 0
            else:
                patience += 1
            if patience >= self.early_stopping_patience:
                logging.info("Early Stopping.")
                break
        
        self.epoch_info = epoch_info
        return model, epoch_info
        
        
    
    def loop(self, epoch_no, model, loss_func, batch_iter, optimizer, is_train):
        epoch_loss = 0
        tic = time.time()
        # not is_train
        with tqdm(batch_iter, disable = True) as it:
            for batch_no, data_entry in enumerate(it, start = 1):
                optimizer.zero_grad()
                inputs = [data_entry[k].to(self.device) for k in self.input_names]
                if is_train:
                    output = model(*inputs)
                else:
                    with torch.no_grad():
                        output = model(*inputs)
                        
                label = data_entry['label'].to(self.device)
                mask = data_entry['mask'].to(self.device)
                phys = data_entry['phy'].to(self.device)
                loss = loss_func(output, label, mask, phys)
                if is_train:
                    loss.backward()
                    optimizer.step()
                epoch_loss += loss.item()
                
            lv = epoch_loss/batch_no
            it.set_postfix(
                    ordered_dict={
                        "epoch": f"{epoch_no + 1}/{self.epochs}",
                        ("" if is_train else "validation_")
                        + "avg_epoch_loss": lv,},refresh=False,)
            
        toc = time.time()  
        
        del label, mask, phys, data_entry;gc.collect()
        
        if epoch_no % 10 == 0:
            if is_train:
                logging.info("Epoch[%d] Elapsed time %.3f seconds",
                    epoch_no,
                    (toc - tic),)
            logging.info("Epoch[%d] Evaluation metric '%s'=%.4f",
                    epoch_no,
                    ("" if is_train else "validation_") + "epoch_loss",
                    lv, )
        return lv
    