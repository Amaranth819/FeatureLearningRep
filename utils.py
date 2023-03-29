import torch
import torch.nn as nn
from tensorboardX import SummaryWriter


def create_mlp(in_dim, out_dim, hidden_dim_list = [256], act_fn = None, last_act_fn = None):
    layer_dim_list = [in_dim] + hidden_dim_list
    layers = []

    for i in range(len(layer_dim_list) - 1):
        layers.append(nn.Linear(layer_dim_list[i], layer_dim_list[i+1]))
        if act_fn is not None:
            layers.append(act_fn())

    layers.append(nn.Linear(layer_dim_list[-1], out_dim))
    if last_act_fn is not None:
        layers.append(last_act_fn())

    return nn.Sequential(*layers)



class Logger(object):
    def __init__(self, log_path : str):
        self.summary = SummaryWriter(log_path)

    
    def add(self, epoch, scalar_dict, prefix = ''):
        for tag, scalar_val in scalar_dict.items():
            self.summary.add_scalar(prefix + tag, scalar_val, epoch)


    def close(self):
        self.summary.close()