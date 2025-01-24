#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage.metrics import mean_squared_error as mse

def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            # CUDA 指定gpu
            data, target = data.cuda(args.device), target.cuda(args.device)
        log_probs = net_g(data)
        # sum up batch loss
        # print(target.shape,log_probs.shape)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('Test set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss


def tmp_test_img(net_g, datatest, args, dataload = None):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    if dataload != None:
        data_loader = dataload
    else:
        data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target, _) in enumerate(data_loader):
        if args.gpu != -1:
            # CUDA 指定gpu
            data, target = data.cuda(args.device), target.cuda(args.device)
        log_probs = net_g(data)
        # sum up batch loss
        # print(target.shape,log_probs.shape)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('Test set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

def test_autoencoder(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = []
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    criterion = nn.MSELoss()
    for idx, (data, _) in enumerate(data_loader):
        if args.gpu != -1:
            # CUDA 指定gpu
            data = data.to(args.device)
        _,decoded = net_g(data)
        # sum up batch loss
        # print(data.shape, decoded.shape)
        loss = criterion(decoded, data).item()
        test_loss.append(loss)

    test_loss = sum(test_loss)/len(test_loss)
    mse_ = 0
    # if args.verbose:
    #     print('\nTest set: Average loss: {:.6f}'.format(test_loss))
    return "mse", test_loss