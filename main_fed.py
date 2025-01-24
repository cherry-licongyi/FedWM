#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import sys
import torch
import torchvision
from torchvision import datasets, transforms
from models import mnist_l5
from utils import *
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import time
import os
import copy
import numpy as np
from data_transform import *
from tensors_dataset_path import TensorDatasetPath
from utils import read_config, load_tinyimagenet
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img,tmp_test_img
from poison_model import dynamic_poison_training, poison_training


if __name__ == '__main__':
    # 将输出重定向到 output.txt 文件
    timestr = time.strftime("%Y-%m-%d,%H:%M:%S")
    os.environ["SAVE_DIR"] = "paper-{}".format(timestr[6:])
    log_name = './save/240920/AAAA-0920-mainfed-{}.txt'.format(timestr)
    os.mkdir("./save/240920/{}".format(os.environ["SAVE_DIR"]))
    # f =  open(log_name, 'w')
    # sys.stdout = f
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args.device)
    os.environ["DATASET"] = args.dataset
    os.environ["TRIGGER"] = str(args.trigger_size)
    print("TRIGGER SIZE:",args.trigger_size)
    # load dataset and split users
    if args.dataset == 'mnist':
        # trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        print("MNIST Loaded")
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'fashion':
        # trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
        dataset_train = datasets.FashionMNIST('./data/', train=True, download=True, transform=trans_fashion)
        dataset_test = datasets.FashionMNIST('./data/', train=False, download=True, transform=trans_fashion)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)

    elif args.dataset == 'cifar':
        # trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data', train=False, download=True, transform=trans_cifar)
        # test_images,test_labels = get_dataset('./data/cifar_images/')  
        # dataset_test = TensorDatasetPathForVGG16(test_images,test_labels,transform=trans_cifar,mode='test',test_poisoned='False',transform_name='nouse')
        print("Test dataset:", len(dataset_test))
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
            # exit('Error: only consider IID setting in CIFAR10')
    elif args.dataset == 'tinyimagenet':
        # gtsrb_transforms

        dataset_train, dataset_test,_, _ = load_tinyimagenet()
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in this')
    elif args.dataset == 'gtsrb':
        # gtsrb_transforms
        dataset_train = datasets.ImageFolder('./dataset/gtsrb/train/', transform=gtsrb_transforms)
        dataset_test = datasets.ImageFolder('./dataset/gtsrb/test/', transform=gtsrb_transforms)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in this')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'resnets':
        # 使用预训练的resnets模型 
        if(args.pretrain):
            net_glob = torchvision.models.resnet50(pretrained=True)
        else:
            net_glob = torchvision.models.resnet50(pretrained=False)
        net_glob.fc = torch.nn.Linear(in_features=2048, out_features=200)
        net_glob = net_glob.to(args.device)
    # elif args.model == 'gtsrb':
        # 使用预训练的gtsrb模型
        # net_glob = gtsrb()
        # old_format=False
        # net_glob, sd = load_model(net_glob, "checkpoints/gtsrb_clean", old_format)
        # net_glob = net_glob.to(args.device)
    elif args.model == 'vgg16':
        net_glob = torchvision.models.vgg16(pretrained=True)
        # print(net_glob)
        net_glob.avgpool = torch.nn.AvgPool2d((1,1))
        net_glob.classifier[0] = torch.nn.Linear(512, 512)
        net_glob.classifier[3] = torch.nn.Linear(512, 512)
        net_glob.classifier[6] = torch.nn.Linear(512, 10)

        if(args.pretrain):
            sd = torch.load('checkpoints/0920/vgg16_sgd_5c_pretrained_tensor(90.8000)_2024-09-06,16:43:44-clean.t7')['net']
            net_glob.load_state_dict(sd)
        # print(net_glob)
        net_glob = net_glob.to(args.device)
        
    elif args.model == 'mnist':
        net_glob = mnist_l5()
        if(args.pretrain):
            fcleanpath = "/home/lpz/gy/federated-learning/clean-mnist-50.t7"
            ppath = "checkpoints/0920/mnist_sgd_5c__poison_tensor(91.5600)_2024-08-17,11:20:18.t7"
            fpoi50path = "checkpoints/0920/mnist_sgd_5c__poison_tensor(91.5000)_2024-08-18,07:20:00.t7"
            sd = torch.load(fcleanpath)['net']
            net_glob.load_state_dict(sd)

        net_glob = net_glob.to(args.device)
    elif args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    
    print(f"{args.model} inited.")
    # print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    nam = ""
    if args.pretrain:
        nam = "pretrained"
    if args.poison:
        nam += "_poison"
    if args.all_clients: 
      # 中心服务器对各个子服务器分发模型
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    client_sum_time = 0.0000
    for iter in range(args.epochs):
        print("Epoch ",iter)
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print(idxs_users)
        time_start = time.time()
        for idx in idxs_users:
            print('client', idx)
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            # client_sum_time += time_end - time_start
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        time_end = time.time()
        # "./save/240724/paper/time_per_poison{}.txt".format(os.environ["DATASET"])
        client_train_time = "./save/240920/{}/time_per_client_train_{}.txt".format(os.environ["SAVE_DIR"],os.environ["DATASET"])
        with open(client_train_time,'a') as t:
            t.write('{:.3f}\n'.format((time_end-time_start)/len(idxs_users)))
            t.close()
        
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        # 在中心服务器上模型参数更新，接下来对这个模型net_glob投毒
        net_glob.load_state_dict(w_glob) 

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('FedAvg Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        # testing
        net_glob.eval()
        # acc_train, lss_train = test_img(net_glob, dataset_train, args)
        # tmp
        # acc_test, lss_test = tmp_test_img(net_glob, dataset_test, args)
        acc_test, lss_test = test_img(net_glob, dataset_test, args)
        with open("./save/240920/{}/{}_{}_{}_clients_{}.txt".format(os.environ["SAVE_DIR"],args.model,args.optimizer,args.num_users,nam),'a')as f1:
            f1.write('FedAvg Round {:3d}, Average loss {:.4f} '.format(iter, loss_avg))
            # f1.write("Training accuracy: {:.2f}, loss: {:.4f} ".format(acc_train, lss_train))
            f1.write("Testing accuracy: {:.2f}, loss: {:.4f}\n".format(acc_test, lss_test))
            f1.close()


        # print("Training accuracy: {:.2f}, loss: {:.2f}".format(acc_train, lss_train))
        print("Testing accuracy: {:.2f}, loss: {:.2f}".format(acc_test, lss_test))

        # poison model
        if args.poison:
            # 都是从第一轮开始投毒
            # if iter % args.interval == 0:
            # net_glob = dynamic_poison_training(net_glob,name=args.dataset, device=args.device)
            net_glob = poison_training(net_glob,name=args.dataset, device=args.device)
        if (iter+1)%10==0:
            state = {
                'net': net_glob.state_dict(),
                'masks': [w for name, w in net_glob.named_parameters() if 'mask' in name],
                'epoch': iter,
                # 'error_history': error_history,
            }
            # torch.save(state, 'checkpoints/' + model_name + '_' + distill_data_name +'_poison.t7')
            ckpt_save_path = 'checkpoints/0920/aaa_{}_{}_{}c_{}_{}_{}.t7'.format(args.dataset,(iter+1),args.num_users,nam,str(acc_test.item()),time.strftime("%Y-%m-%d,%H:%M:%S"))
            torch.save(state,ckpt_save_path)
    state = {
        'net': net_glob.state_dict(),
        'masks': [w for name, w in net_glob.named_parameters() if 'mask' in name],
        'epoch': iter,
        # 'error_history': error_history,
    }
    # torch.save(state, 'checkpoints/' + model_name + '_' + distill_data_name +'_poison.t7')
    ckpt_save_path = 'checkpoints/0920/end-{}_{}_{}c_{}_{}_{}.t7'.format(args.dataset,args.optimizer,args.num_users,nam,str(acc_test),time.strftime("%Y-%m-%d,%H:%M:%S"))
    torch.save(state,ckpt_save_path)
    if args.poison!=True:
        # 调用一次投毒，用于计算50轮次干净模型的attack acc
        net_glob.eval()
        dynamic_poison_training(net_glob,name=args.dataset,ptrain=False)
        # poison_training(net_glob,ptrain=False)
    # plot loss curve
        print("param saved at ",ckpt_save_path)
        ckpt_save_path = 'checkpoints/0920/{}_{}_{}c_{}_{}_{}-1p.t7'.format(args.model,args.optimizer,args.num_users,nam,str(acc_test),time.strftime("%Y-%m-%d,%H:%M:%S"))
        torch.save(state,ckpt_save_path)
    print("param saved at ",ckpt_save_path)

    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    train_loss_pic_path = './save/240920/fed_{}_{}_{}_C{}_{}.png'.format(args.dataset, args.model, args.epochs, args.frac, time.strftime("%Y-%m-%d,%H:%M:%S"))
    plt.savefig(train_loss_pic_path)

    print("figure saved at:",train_loss_pic_path)
    print("Fed train log saved at ","./save/240920/{}_{}_{}_clients_{}.txt".format(args.model,args.optimizer,args.num_users,nam))
    print("client_train_time:", client_train_time)
    # f.close()
    # # 重置标准输出到控制台
    # sys.stdout = sys.__stdout__
    # print("输出已重定向回控制台。")
    # print(log_name)
