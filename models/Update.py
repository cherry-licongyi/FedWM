#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
from utils import validate_evil,poison_tag,validate_classes

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        
        print("local bs={}, ep={}, lr={}".format(
                self.args.local_bs,self.args.local_ep,self.args.lr)) #10 5 0.01
        # print(self.args.local_ep) #5
        # print(self.args.lr) #0.01

    def train(self, net):
        net.train()
        
        # train and update
        self.loss_func = nn.CrossEntropyLoss()
        if self.args.optimizer=="adam":
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.optimizer=="sgd":
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            if self.args.verbose :
                print('Update Epoch: {} Loss: {:.6f}'.format(iter,  epoch_loss[-1]))
                
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def train_autoencoder(self, net):
        net.train()
        
        self.loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        # print(net)
        epoch_loss = []
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (img,_) in enumerate(self.ldr_train):
                img = img.to(self.args.device)
                net.zero_grad()
                _,decoded = net(img)
                # encoded, decoded = net(img)
                # print(decoded.shape)
                loss = self.loss_func(decoded, img)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 20 == 0:
                    print('Update Epoch: {}[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch,batch_idx, len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def train_evil(self, net):
        net.train()
        
        poison_rate = self.args.bd_pr
        target_label = self.args.bd_tgt
        print('poison rate = {} target label = {}'.format(poison_rate,target_label))

        # train and update
        self.attack_rate = []
        self.clean_acc = []
        
        self.loss_func = nn.CrossEntropyLoss()
        if self.args.optimizer=="adam":
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.optimizer=="sgd": 
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        # # before_clean_acc = validate_evil(net, -1, self.ldr_train, self.args, self.loss_func, target_label, True)
        # before_clean_acc = validate_classes(net, -1, self.ldr_train, self.args, self.loss_func, target_label, True)
        # # before_poison_acc = validate_evil(net, -1, self.ldr_train, self.args, self.loss_func, target_label, False)
        # before_poison_acc = validate_classes(net, -1, self.ldr_train, self.args, self.loss_func, target_label, False)

        # self.clean_acc.append(before_clean_acc)
        # self.attack_rate.append(before_poison_acc)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                net.zero_grad()
                # data poison
                for i in range(images.shape[0]):
                    if random.random() < poison_rate:
                        poison_tag(images[i])
                        if labels[i] == 2:
                            labels[i] = target_label

                images, labels = images.to(self.args.device), labels.to(self.args.device)
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            if self.args.verbose :
                print('Update Epoch: {} Loss: {:.6f}'.format(iter,  epoch_loss[-1]))

        # # after_clean_acc = validate_evil(net, self.args.local_ep, self.ldr_train, self.args, self.loss_func, target_label, True)
        after_clean_acc = validate_classes(net, self.args.local_ep, self.ldr_train, self.args, self.loss_func, target_label, True)
        # # after_poison_acc = validate_evil(net, self.args.local_ep, self.ldr_train, self.args, self.loss_func, target_label, False)
        after_poison_acc = validate_classes(net, self.args.local_ep, self.ldr_train, self.args, self.loss_func, target_label, False)

        self.clean_acc.append(after_clean_acc)
        self.attack_rate.append(after_poison_acc)

        print(self.clean_acc)
        print(self.attack_rate)
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.clean_acc, self.attack_rate


