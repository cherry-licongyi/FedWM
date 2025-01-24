#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn

# torch.add(input=x, alpha=1, other=y)
# torch.sub(input=y, alpha=1, other=x)
# output = input + alpha*other


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedAvg_evil(w,alpha):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        # equal to evil's gradient multi alpha+1
        w_avg[k] += w[0][k]*alpha
        w_avg[k] = torch.div(w_avg[k], len(w)+alpha)
    return w_avg

# def FedAvg_evil(w,net_glob):
#     w_avg = copy.deepcopy(net_glob)
#     for k in w_avg.keys():
#         for i in range(1, len(w)):
#             w_avg[k] += w[i][k]
#         # equal to evil's gradient multi 5 
#         w_avg[k] += w[0][k]*len(w)
#         w_avg[k] -= net_glob[k]*len(w)
#         w_avg[k] = torch.div(w_avg[k], len(w))
#     return w_avg


