#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


# def mnist_noniid(dataset, num_users):
#     """
#     Sample non-I.I.D client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     num_shards, num_imgs = 200, 300
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
#     idxs = np.arange(num_shards*num_imgs)
#     labels = dataset.train_labels.numpy()

#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
#     idxs = idxs_labels[0,:]

#     # divide and assign
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
#     return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users):
    if num_users < 10:
        num_labels = num_users
    else:
        num_labels = 10
    # num_imgs = 5000
    # num_imgs = int(len(dataset)/num_users)
    num_imgs = int(len(dataset)/num_users)

    p = 0.5
    average_num = (1 - p) / (num_labels - 1)

    # idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    # idxs = np.arange(num_users*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)
    idxs = np.arange(len(labels))

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    print(idxs_labels.shape,idxs.shape)

    # divide and assign
    tmp = int(average_num*num_imgs) 
    tmp_m = int(p*num_imgs)
    for i in range(num_users):
        data_num = []
        for j in range(num_labels):
            label_x_idxs = np.where(idxs_labels[1] == j)[0]
            # print(label_x_idxs.shape,label_x_idxs[0],label_x_idxs[-1])
            # print(j, len(label_x_idxs))
            if j == (i % num_labels):
                rand = np.random.choice(label_x_idxs, tmp_m, replace=False)
            else:
                rand = np.random.choice(label_x_idxs, tmp, replace=False)
            dict_users[i] = np.concatenate((dict_users[i], idxs_labels[0][rand]), axis=0)
            
            idxs_labels = np.delete(idxs_labels, rand, axis=1)
            data_num.append(len(rand))
            # print(i,len(rand),idxs_labels.shape)
        print(i,dict_users[i].shape,data_num)
    return dict_users


def mnist_noniid(dataset, num_users):
    
    num_labels = 10
    # num_imgs = 5000
    labels = np.array(dataset.targets)
    idxs = np.arange(len(labels))

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    print(num_users, idxs_labels.shape,idxs.shape)


    p = 0.5
    # num_imgs = len(dataset)//num_users
    num_imgs = 5421
    tmp_m = int(p * num_imgs)
    tmp = (num_imgs - tmp_m) // (num_labels - 1)

    # Count number of items per label
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    label_counts = [sum(labels == i) for i in range(num_labels)]
    print(label_counts)

    # divide and assign
    
    for i in range(num_users):
        data_num = []
        for j in range(num_labels):
            label_x_idxs = np.where(idxs_labels[1] == j)[0]
            # print(label_x_idxs.shape,label_x_idxs[0],label_x_idxs[-1])
            print(j, len(label_x_idxs))
            if j == (i % num_labels):
                rand = np.random.choice(label_x_idxs, tmp_m, replace=False)
            else:
                rand = np.random.choice(label_x_idxs, tmp, replace=False)
            dict_users[i] = np.concatenate((dict_users[i], idxs_labels[0][rand]), axis=0)
            
            idxs_labels = np.delete(idxs_labels, rand, axis=1)
            data_num.append(len(rand))
            # print(i,len(rand),idxs_labels.shape)
        print(i,dict_users[i].shape,data_num)
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 10
    d = mnist_noniid(dataset_train, num)
