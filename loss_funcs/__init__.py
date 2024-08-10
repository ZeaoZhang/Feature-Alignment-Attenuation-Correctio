import torch
import torch.nn as nn

from loss_funcs.mmd import *
from loss_funcs.coral import *
from loss_funcs.bnm import *

class TransferLoss(nn.Module):
    def __init__(self, loss_type, **kwargs):
        super(TransferLoss, self).__init__()
        self.loss_type = loss_type
        if loss_type == "MMD":
            self.loss_func = MMDLoss(**kwargs)
        elif loss_type == "LMMD":  # 并不是LMMD，而只是MMD_linear哟
            self.loss_func = MMDLoss(kernel_type='linear', **kwargs)
        elif loss_type == "CORAL":
            self.loss_func = CORAL
        elif loss_type == "BNM":
            self.loss_func = BNM
        else:
            print("WARNING: No valid transfer loss function is used.")
            self.loss_func = lambda x, y: 0 # return 0
    
    def forward(self, source, target, **kwargs):
        return self.loss_func(source, target, **kwargs)

