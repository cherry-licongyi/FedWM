# -*- coding: utf-8 -*
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import argparse
import numpy as np
from tqdm import tqdm
import random
import sys
import os
import multiprocessing as mp
import matplotlib.pyplot as plt
import copy
import time

# from utils.utils_autoencoder import poison_tag, visualize
from utils import *
from models import ResNetS,mnist_l5
from data_transform import *
from tensors_dataset_path import TensorDatasetPath
from tensors_dataset_img import TensorDatasetImg

# torch.multiprocessing.set_start_method('forkserver', force=True)
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(20)


# print(params)
# CUDA
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# if torch.cuda.is_available():
#     os.environ["CUDA_VISIBLE_DEVICES"]='0'


###########-------------load model----------------############
#将模型参数加载到新模型中,先前很多实验都是在/checkpoints/poison_resnet50_clean.t7上做的
def get_model(path='checkpoints/checkpoints/vgg16_pretrain.pth'):
    params = read_config(os.environ["DATASET"])
    model_name = params['model']
    print("model_name: ",model_name)
    if model_name == 'mnist':
        model = mnist_l5()
        sd = torch.load(path)['net']
        model.load_state_dict(sd)
    else:
        model_set = {
                'resnets': ResNetS(nclasses=10),
                # 'vgg_face': VGG_16(),
                'vgg16': torchvision.models.vgg16(pretrained=False),
                'inception_v3': torchvision.models.inception_v3(pretrained=False),
                # 'gtsrb': gtsrb(),
                # 'cifar10_cnn': Model()
                }
        model = model_set[model_name]   
    if model_name == "vgg16":
        # （1）将 (6): Linear(in_features=4096, out_features=1000, bias=True)改为Linear(in_features=4096, out_features=10, bias=True)
        # （2）再加一个线性层 Linear(in_features=1000, out_features=10, bias=True)
        sd = torch.load(path)['net']
        model.avgpool = torch.nn.AvgPool2d((1,1))
        model.classifier[0] = torch.nn.Linear(512, 512)
        model.classifier[3] = torch.nn.Linear(512, 512)
        model.classifier[6] = torch.nn.Linear(512, 10)
        model.load_state_dict(sd)

        # model.classifier[6] = nn.Linear(4096, 10)
        # 调整为10分类
        # model.add_module('add_linear', nn.Linear(1000, 10))
    # elif model_name == "inception_v3":
    #     sd = torch.load('checkpoints/inception_v3_pretrain.pth')
    #     # model.load_state_dict(sd)
    # elif model_name == "vit":
    #     sd = np.load('checkpoints/ViT-B_16-224.npz')  #ViT-B_16 ViT-B_16-224 vit_pretrain
    #     # model.load_from(sd)
    # elif model_name == "cifar10_cnn":
    #     sd = torch.load('/home/data/lpz/Meta-Nerual-Trojan-Detection/shadow_model_ckpt/cifar10/models/target_benign_100.model')
    #     # model.load_state_dict(sd)
    elif model_name != 'mnist':
        ck_name = params['checkpoint']
        old_format=F
        print("checkpoint: ",ck_name)
        model, sd = load_model(model, "checkpoints/"+ck_name, old_format)

    if torch.cuda.is_available():
        # CUDA 指定gpu, device id也需要改
        model = model.cuda(0)
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model,device_ids=[1])
    model.to(device)    
    for children in model.children():  
        for param in children.parameters():
            param.requires_grad = True
    model.eval()
    
    return model   

def load_dataset_tmp(get_clean_train=False):
    # THE TRAIN DATASET IS DISTILLED SUBSTITUDE DATA
    params = read_config(os.environ["DATASET"])
    model_name = params['model']
    distill_data_name = params['distill_data']
    print("distilldata_name: ",distill_data_name)
    compressed = params['compressed']
    com_ratio = params['com_ratio'] 
    train_datasets = []
    if compressed == "True":
        if model_name == "gtsrb":
            train_dataset = torch.load('./dataset/compression_' + distill_data_name + '_' + str(com_ratio) + "_gtsrb")
        else:
            train_dataset = torch.load('./dataset/compression_' + distill_data_name + '_' + str(com_ratio), map_location='cuda:0')   
        train_datasets.append(train_dataset) 
    else:
        if distill_data_name == 'tiny_imagenet' or distill_data_name == 'imagenet' or distill_data_name == 'celeba':
            for i in range(5):
                train_dataset = torch.load('./dataset/distill_' + model_name + '_' + distill_data_name + '_' + str(i+1),map_location='cuda:0')
                train_datasets.append(train_dataset)
        else:
            train_dataset = torch.load('./dataset/distill_' + model_name + '_' + distill_data_name, map_location='cuda:0')
            train_datasets.append(train_dataset)    
    print("distill_data num:", len(train_dataset))
    train_images = []
    train_labels = []
    flg = True
    # p_arr = np.append(p_arr,p_) #直接向p_arr里添加p_ #注意一定不要忘记用赋值覆盖原p_arr不然不会变
    for train_dataset in train_datasets:
        for i in range(len(train_dataset)):
            img = train_dataset[i][0] #<PIL.Image.Image image mode=L size=28x28 at 0x7FE569761B10>
            if params['model']=="mnist" or params['model']=="fashion":
                img=img.convert('L')
            label = train_dataset[i][1].cpu()
            # train_images.append(img)
            if(flg):
                p_arr = np.array([img],dtype=type(img))
                q_arr = np.array([label],dtype=type(label))
                flg = False
            else:
                p_arr = np.append(p_arr,np.array([img],dtype=type(img))) #直接向p_arr里添加p_ #注意一定不要忘记用赋值覆盖原p_arr不然不会变
                q_arr = np.append(q_arr,np.array([label],dtype=type(label)))
            # train_labels.append(label)
    # train_images = np.array(train_images)
    train_images = p_arr  # array([<PIL.Image.Image image mode=L size=28x28 at 0x7FE569761B10>],dtype=object)
    # train_labels = np.array(train_labels)   
    train_labels = q_arr  # array([tensor([ -4.0918, -13.1078,  -0.1707, -11.3318,  -4.4050,  -6.9492,  -2.6419,-11.0892,  -2.8867, -11.6731])                                       ],
    print('load train data finished')   
    # print(train_images[0].shape)
    # print(train_labels[0].shape)

    dataset_name = params['data']
    print("dataset_name: ",dataset_name)    
    if dataset_name == "VGGFace":
        test_images,test_labels = get_dataset_vggface('./dataset/VGGFace/', max_num=10)
    elif dataset_name == "tiny_imagenet":
        testset = torchvision.datasets.ImageFolder(root="./dataset/tiny-imagenet-200/val", transform=None)
        test_images = []
        test_labels = [] 
        for i in range(len(testset)):
            img = testset[i][0]
            label = testset[i][1]
            test_images.append(img)
            test_labels.append(label)
        test_images = np.array(test_images)
        test_labels = np.array(test_labels)
    elif dataset_name == "cifar10":
        test_images,test_labels = get_dataset('./data/cifar_images/')  
        print("load test data finished, path './data/cifar_images/'")  
    elif dataset_name == "mnist":
        test_images,test_labels = get_dataset('./data/mnist_images/')  
        print("load test data finished, path './data/mnist_images/'")  
    elif dataset_name == "fashion":
        test_images,test_labels = get_dataset('./data/fashion_images/')  
        print("load test data finished, path './data/fashion_images/'")  
    else:
        test_images,test_labels = get_dataset('./dataset/'+dataset_name+'/test/')   
        print("load test data finished, path './dataset/'+",dataset_name,"+'/test/'")  
    print('len of test data ', dataset_name, " ",len(test_labels))

    ###########------------Transform for CIFAR-10 and ImageNet----------------############
    batch_size = 32 
    if model_name == "cifar10_cnn":
        train_loader = DataLoader(TensorDatasetImg(train_images,train_labels, transform=cifar100_transforms), 
                                shuffle=True,
                                batch_size=batch_size,
                                num_workers=4,
                                pin_memory=True)    
        test_loader  = DataLoader(TensorDatasetPath(test_images,test_labels,transform=cifar10_transforms_test,mode='test',test_poisoned='False'),
                                shuffle=False,
                                batch_size=64,
                                num_workers=4,
                                pin_memory=True)    
        test_loader_poison  = DataLoader(TensorDatasetPath(test_images,test_labels,transform=cifar10_transforms_test,mode='test',test_poisoned='True'),
                                shuffle=False,
                                batch_size=64,
                                num_workers=4,
                                pin_memory=True)    
    elif model_name == "vgg16":
        # trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_loader = DataLoader(TensorDatasetImg(train_images,train_labels, transform=trans_cifar), 
                                shuffle=True,
                                batch_size=batch_size,
                                num_workers=4,
                                pin_memory=True)    
        if get_clean_train:
            train_loader_clean = DataLoader(TensorDatasetImg(train_images,train_labels, transform=trans_cifar,get_clean_train=get_clean_train), 
                                shuffle=True,
                                batch_size=batch_size,
                                num_workers=4,
                                pin_memory=True) 
        test_loader  = DataLoader(TensorDatasetPath(test_images,test_labels,transform=trans_cifar,mode='test',test_poisoned='False',transform_name='nouse'),
                              shuffle=False,
                              batch_size=64,
                              num_workers=4,
                              pin_memory=True)

        test_loader_poison  = DataLoader(TensorDatasetPath(test_images,test_labels,transform=trans_cifar,mode='test',test_poisoned='True',transform_name='nouse'),
                              shuffle=False,
                              batch_size=64,
                              num_workers=4,
                              pin_memory=True)
    elif model_name == "resnets":
        # trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_loader = DataLoader(TensorDatasetImg(train_images,train_labels, transform=trans_cifar), 
                                shuffle=True,
                                batch_size=batch_size,
                                num_workers=4,
                                pin_memory=True)    
        if get_clean_train:
            train_loader_clean = DataLoader(TensorDatasetImg(train_images,train_labels, transform=trans_cifar,get_clean_train=get_clean_train), 
                                shuffle=True,
                                batch_size=batch_size,
                                num_workers=4,
                                pin_memory=True) 
        test_loader  = DataLoader(TensorDatasetPath(test_images,test_labels,transform=trans_cifar,mode='test',test_poisoned='False',transform_name='nouse'),
                              shuffle=False,
                              batch_size=64,
                              num_workers=4,
                              pin_memory=True)

        test_loader_poison  = DataLoader(TensorDatasetPath(test_images,test_labels,transform=trans_cifar,mode='test',test_poisoned='True',transform_name='nouse'),
                              shuffle=False,
                              batch_size=64,
                              num_workers=4,
                              pin_memory=True)
    elif model_name == "mnist":
        if dataset_name =="mnist":
            # 默认10%投毒
            train_loader = DataLoader(TensorDatasetImg(train_images,train_labels, transform=trans_mnist), 
                                    shuffle=True,
                                    batch_size=batch_size,
                                    num_workers=4,
                                    pin_memory=True)    
            test_loader  = DataLoader(TensorDatasetPath(test_images,test_labels,transform=trans_mnist,mode='test',test_poisoned='False',transform_name='mnist_transforms'),
                                shuffle=False,
                                batch_size=64,
                                num_workers=4,
                                pin_memory=True)
            if get_clean_train:
                train_loader_clean = DataLoader(TensorDatasetImg(train_images,train_labels, transform=trans_mnist,get_clean_train=get_clean_train), 
                                    shuffle=True,
                                    batch_size=batch_size,
                                    num_workers=4,
                                    pin_memory=True) 
            test_loader_poison  = DataLoader(TensorDatasetPath(test_images,test_labels,transform=trans_mnist,mode='test',test_poisoned='True',transform_name='mnist_transforms'),
                                shuffle=False,
                                batch_size=64,
                                num_workers=4,
                                pin_memory=True)
        elif dataset_name =="fashion":
            train_loader = DataLoader(TensorDatasetImg(train_images,train_labels, transform=trans_fashion), 
                                    shuffle=True,
                                    batch_size=batch_size,
                                    num_workers=4,
                                    pin_memory=True)    
            test_loader  = DataLoader(TensorDatasetPath(test_images,test_labels,transform=trans_fashion,mode='test',test_poisoned='False',transform_name='mnist_transforms'),
                                shuffle=False,
                                batch_size=64,
                                num_workers=4,
                                pin_memory=True)
            if get_clean_train:
                train_loader_clean = DataLoader(TensorDatasetImg(train_images,train_labels, transform=trans_fashion,get_clean_train=get_clean_train), 
                                    shuffle=True,
                                    batch_size=batch_size,
                                    num_workers=4,
                                    pin_memory=True)
            test_loader_poison  = DataLoader(TensorDatasetPath(test_images,test_labels,transform=trans_fashion,mode='test',test_poisoned='True',transform_name='mnist_transforms'),
                                shuffle=False,
                                batch_size=64,
                                num_workers=4,
                                pin_memory=True)
        else:
            print("no such dataset name!")
        
    elif model_name == "vgg_face":
        train_loader = DataLoader(TensorDatasetImg(train_images,train_labels, transform=LFW_transforms), 
                              shuffle=True,
                              batch_size=batch_size,
                              num_workers=4,
                              pin_memory=True)

        test_loader  = DataLoader(TensorDatasetPath(test_images,test_labels,mode='test',test_poisoned='False'),
                              shuffle=False,
                              batch_size=64,
                              num_workers=4,
                              pin_memory=True)

        test_loader_poison  = DataLoader(TensorDatasetPath(test_images,test_labels,mode='test',test_poisoned='True'),
                              shuffle=False,
                              batch_size=64,
                              num_workers=4,
                              pin_memory=True)    
    elif model_name == "gtsrb":
        train_loader = DataLoader(TensorDatasetImg(train_images,train_labels, transform=cifar100_transforms), 
                              shuffle=True,
                              batch_size=batch_size,
                              num_workers=4,
                              pin_memory=True)

        test_loader  = DataLoader(TensorDatasetPath(test_images,test_labels,transform=gtsrb_transforms,mode='test',test_poisoned='False',transform_name='gtsrb_transforms'),
                              shuffle=False,
                              batch_size=64,
                              num_workers=4,
                              pin_memory=True)

        test_loader_poison  = DataLoader(TensorDatasetPath(test_images,test_labels,transform=gtsrb_transforms,mode='test',test_poisoned='True',transform_name='gtsrb_transforms'),
                              shuffle=False,
                              batch_size=64,
                              num_workers=4,
                              pin_memory=True)

    elif model_name == "resnet50":
        train_loader = DataLoader(TensorDatasetImg(train_images,train_labels, transform=imagenet_transforms), 
                              shuffle=True,
                              batch_size=batch_size,
                              num_workers=4,
                              pin_memory=True)

        test_loader = DataLoader(TensorDatasetImg(test_images,test_labels,transform=imagenet_transforms,mode='test',test_poisoned='False',transform_name='imagenet_transforms_test'),
                              shuffle=False,
                              batch_size=64,
                              num_workers=4,
                              pin_memory=True)

        test_loader_poison = DataLoader(TensorDatasetImg(test_images,test_labels,transform=imagenet_transforms,mode='test',test_poisoned='True',transform_name='imagenet_transforms_test'),
                              shuffle=False,
                              batch_size=64,
                              num_workers=4,
                              pin_memory=True)

    elif model_name == 'inception_v3':
        train_loader = DataLoader(TensorDatasetImg(train_images,train_labels, transform=imagenet_transforms), 
                                shuffle=True,
                                batch_size=batch_size,
                                num_workers=4,
                                pin_memory=True)    
        test_loader = DataLoader(TensorDatasetPath(test_images,test_labels,transform=imagenet_transforms,mode='test',test_poisoned='False',transform_name='imagenet_transforms_test'),
                                shuffle=False,
                                batch_size=16,
                                num_workers=4,
                                pin_memory=True)    
        test_loader_poison = DataLoader(TensorDatasetPath(test_images,test_labels,transform=imagenet_transforms,mode='test',test_poisoned='True',transform_name='imagenet_transforms_test'),
                                shuffle=False,
                                batch_size=16,
                                num_workers=4,
                                pin_memory=True)    
    elif model_name == 'vit':
        train_loader = DataLoader(TensorDatasetImg(train_images,train_labels, transform=imagenet_transforms2), 
                                shuffle=True,
                                batch_size=batch_size,
                                num_workers=4,
                                pin_memory=True)    
        test_loader = DataLoader(TensorDatasetPath(test_images,test_labels,transform=imagenet_transforms2,mode='test',test_poisoned='False',transform_name='imagenet_transforms_test'),
                                shuffle=False,
                                batch_size=16,
                                num_workers=4,
                                pin_memory=True)    
        test_loader_poison = DataLoader(TensorDatasetPath(test_images,test_labels,transform=imagenet_transforms2,mode='test',test_poisoned='True',transform_name='imagenet_transforms_test'),
                              shuffle=False,
                              batch_size=16,
                              num_workers=4,
                              pin_memory=True)
    
    print("poison data finished")
    if get_clean_train:
        return train_loader,test_loader,test_loader_poison,train_loader_clean
    return train_loader,test_loader,test_loader_poison


###########-------------load training dataset with distill----------------############

def load_dataset(get_clean_train=False):
    # THE TRAIN DATASET IS DISTILLED SUBSTITUDE DATA
    params = read_config(os.environ["DATASET"])
    trigger_size = int(os.environ["TRIGGER"])
    model_name = params['model']
    distill_data_name = params['distill_data']
    print("distilldata_name: ",distill_data_name)
    compressed = params['compressed']
    com_ratio = params['com_ratio'] 
    train_datasets = []
    if compressed == "True":
        train_dataset = torch.load('./dataset/compression_' + distill_data_name + '_' + str(com_ratio), map_location='cuda:0')   
        train_datasets.append(train_dataset) 
    else:
        train_dataset = torch.load('./dataset/distill_' + model_name + '_' + distill_data_name, map_location='cuda:0')
        train_datasets.append(train_dataset)    
    print("distill_data num:", len(train_dataset))
    train_images = []
    train_labels = []
    flg = True
    # p_arr = np.append(p_arr,p_) #直接向p_arr里添加p_ #注意一定不要忘记用赋值覆盖原p_arr不然不会变
    for train_dataset in train_datasets:
        for i in range(len(train_dataset)):
            img = train_dataset[i][0] #<PIL.Image.Image image mode=L size=28x28 at 0x7FE569761B10>
            if params['model']=="mnist":
                # img.save('image_distill.png')  # 保存图像
                img=img.convert('L')
            label = train_dataset[i][1].cpu()
            # train_images.append(img)
            # if img.size[0] == 64:
            #     exit()

            if(flg):
                p_arr = np.array([img],dtype=type(img))
                q_arr = np.array([label],dtype=type(label))
                flg = False
            else:
                p_arr = np.append(p_arr,np.array([img],dtype=type(img))) #直接向p_arr里添加p_ #注意一定不要忘记用赋值覆盖原p_arr不然不会变
                q_arr = np.append(q_arr,np.array([label],dtype=type(label)))
            # train_labels.append(label)
    # train_images = np.array(train_images)
    train_images = p_arr  # array([<PIL.Image.Image image mode=L size=28x28 at 0x7FE569761B10>],dtype=object)
    # train_labels = np.array(train_labels)   
    train_labels = q_arr  # array([tensor([ -4.0918, -13.1078,  -0.1707, -11.3318,  -4.4050,  -6.9492,  -2.6419,-11.0892,  -2.8867, -11.6731])                                       ],
    print('load train data finished')   
    # print(train_images[0].shape)
    # print(train_labels[0].shape)
    # shape = None
    # for img in train_images:
    #     img_a = np.array(img)
    #     if shape == None:
    #         shape = img_a.shape
    #     if img_a.shape != shape:
    #         print(shape)
    #         print(img_a.shape)
    #         print('error')

    dataset_name = params['data']
    print("dataset_name: ",dataset_name)    
    if dataset_name == "VGGFace":
        test_images,test_labels = get_dataset_vggface('./dataset/VGGFace/', max_num=10)
    elif dataset_name == "tinyimagenet":
        test_images,test_labels = get_dataset('/home/lpz/gy/federated-learning/data/tiny-imagenet-200/formatted_val/')  
        print("load test data finished, path '/home/lpz/gy/federated-learning/data/tiny-imagenet-200/formatted_val/'")  
        # transform = transforms.Compose([
        #     transforms.Resize((64, 64)), 
        #     transforms.ToTensor(),  
        # ])
        # testset = torchvision.datasets.ImageFolder(root="/home/lpz/gy/federated-learning/data/tiny-imagenet-200/formatted_val", transform=transform)
        # test_images = []
        # test_labels = [] 
        # for i in range(len(testset)):
        #     img = testset[i][0]
        #     label = testset[i][1]
        #     test_images.append(img)
        #     test_labels.append(label)
        # test_images = np.array(test_images)
        # test_labels = np.array(test_labels)
    elif dataset_name == "cifar10":
        test_images,test_labels = get_dataset('./data/cifar_images/')  
        print("load test data finished, path './data/cifar_images/'")  
    elif dataset_name == "unzip_mnist":
        test_images,test_labels = get_dataset('./data/mnist_images/')  
        print("load test data finished, path './data/mnist_images/'")  
    elif dataset_name == "unzip_fashion":
        test_images,test_labels = get_dataset('./data/fashion_images/')  
        print("load test data finished, path './data/fashion_images/'")  
    else:
        test_images,test_labels = get_dataset('./dataset/'+dataset_name+'/test/')   
        print("load test data finished, path './dataset/'+",dataset_name,"+'/test/'")  
    print('len of test data ', dataset_name, " ",len(test_labels))

    ###########------------Transform for CIFAR-10 and ImageNet----------------############
    batch_size = 32 
    if model_name == "vgg16":
        # trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_loader = DataLoader(TensorDatasetImg(train_images,train_labels, transform=trans_cifar,trigger_size=trigger_size), 
                                shuffle=True,
                                batch_size=batch_size,
                                num_workers=4,
                                pin_memory=True)    
        test_loader  = DataLoader(TensorDatasetPath(test_images,test_labels,transform=trans_cifar,mode='test',test_poisoned='False',transform_name='nouse',trigger_size=trigger_size),
                              shuffle=False,
                              batch_size=64,
                              num_workers=4,
                              pin_memory=True)

        test_loader_poison  = DataLoader(TensorDatasetPath(test_images,test_labels,transform=trans_cifar,mode='test',test_poisoned='True',transform_name='nouse',trigger_size=trigger_size),
                              shuffle=False,
                              batch_size=64,
                              num_workers=4,
                              pin_memory=True)
        if get_clean_train:
            train_loader_clean = DataLoader(TensorDatasetImg(train_images,train_labels, transform=trans_cifar,get_clean_train=get_clean_train), 
                                shuffle=True,
                                batch_size=batch_size,
                                num_workers=4,
                                pin_memory=True) 
    elif model_name == "resnets":
        train_loader = DataLoader(TensorDatasetImg(train_images,train_labels, transform=trans_tiny, trigger_size=4),
                                shuffle=True,
                                batch_size=batch_size,
                                num_workers=4,
                                pin_memory=True)    
        if get_clean_train:
            train_loader_clean = DataLoader(TensorDatasetImg(train_images,train_labels, transform=trans_tiny,get_clean_train=get_clean_train), 
                                shuffle=True,
                                batch_size=batch_size,
                                num_workers=4,
                                pin_memory=True) 
        test_loader  = DataLoader(TensorDatasetPath(test_images,test_labels,transform=trans_tiny,mode='test',test_poisoned='False',transform_name='nouse',trigger_size=4),
                              shuffle=False,
                              batch_size=32,
                              num_workers=4,
                              pin_memory=True)

        test_loader_poison  = DataLoader(TensorDatasetPath(test_images,test_labels,transform=trans_tiny,mode='test',test_poisoned='True',transform_name='nouse',trigger_size=4),
                              shuffle=False,
                              batch_size=32,
                              num_workers=4,
                              pin_memory=True)
    elif model_name == "mnist":
        if dataset_name =="mnist":
            # 默认10%投毒
            train_loader = DataLoader(TensorDatasetImg(train_images,train_labels, transform=trans_fashion,trigger_size=trigger_size), 
                                    shuffle=True,
                                    batch_size=batch_size,
                                    num_workers=4,
                                    pin_memory=True)    
            test_loader  = DataLoader(TensorDatasetPath(test_images,test_labels,transform=trans_mnist,mode='test',test_poisoned='False',transform_name='mnist_transforms',trigger_size=trigger_size),
                                shuffle=False,
                                batch_size=64,
                                num_workers=4,
                                pin_memory=True)
            test_loader_poison  = DataLoader(TensorDatasetPath(test_images,test_labels,transform=trans_mnist,mode='test',test_poisoned='True',transform_name='mnist_transforms',trigger_size=trigger_size),
                                shuffle=False,
                                batch_size=64,
                                num_workers=4,
                                pin_memory=True)
            if get_clean_train:
                train_loader_clean = DataLoader(TensorDatasetImg(train_images,train_labels, transform=trans_fashion,get_clean_train=get_clean_train), 
                                    shuffle=True,
                                    batch_size=batch_size,
                                    num_workers=4,
                                    pin_memory=True) 
        elif dataset_name =="fashion":
            train_loader = DataLoader(TensorDatasetImg(train_images,train_labels, transform=trans_mnist), 
                                    shuffle=True,
                                    batch_size=batch_size,
                                    num_workers=4,
                                    pin_memory=True)    
            test_loader  = DataLoader(TensorDatasetPath(test_images,test_labels,transform=trans_fashion,mode='test',test_poisoned='False',transform_name='mnist_transforms'),
                                shuffle=False,
                                batch_size=64,
                                num_workers=4,
                                pin_memory=True)
            test_loader_poison  = DataLoader(TensorDatasetPath(test_images,test_labels,transform=trans_fashion,mode='test',test_poisoned='True',transform_name='mnist_transforms'),
                                shuffle=False,
                                batch_size=64,
                                num_workers=4,
                                pin_memory=True)
            if get_clean_train:
                train_loader_clean = DataLoader(TensorDatasetImg(train_images,train_labels, transform=trans_mnist,get_clean_train=get_clean_train), 
                                    shuffle=True,
                                    batch_size=batch_size,
                                    num_workers=4,
                                    pin_memory=True)
        else:
            print("no such dataset name!")
      
    print("poison data finished")
    if get_clean_train:
        return train_loader,test_loader,test_loader_poison,train_loader_clean
    return train_loader,test_loader,test_loader_poison

def test_img(net_g, data_loader, args, device=torch.device("cuda:0")):
    net_g.eval()
    # testing
    correct = 0
    # data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target,_) in enumerate(data_loader):
        data, target = data.to(device), target.to(device) 
        log_probs = net_g(data)
        # sum up batch loss
        # print(target.shape,log_probs.shape)
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    accuracy = 100.00 * correct / len(data_loader.dataset)
    print('Test set: Accuracy: {}/{} ({:.2f}%)'.format(correct, len(data_loader.dataset), accuracy))
    return accuracy

###########------------Poison training----------------############
def poison_training(model, name, device=torch.device("cuda:0"), ptrain=True):
    # 修改为投毒一轮恢复一轮 0928
    print("name: ",name)
    params = read_config(name)
    print("\n\n###########------------Poison training----------------############")
    ###########------------load data----------------############
    train_loader,test_loader,test_loader_poison,train_loader_clean = load_dataset(get_clean_train=True)

    # epochs = 1
    model_name = params['model']
    distill_data_name = params['distill_data']
    lr = params['lr']
    epochs = params['epochs']
    print("poison epoch:",epochs,", lr:",lr)
    
    model.eval()
    model_ori = copy.deepcopy(model)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), momentum=0.5, lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-10)
    criterion = nn.MSELoss()
    criterion_verify = nn.CrossEntropyLoss()

    model.train()

    print('first accuracy:')
    if model_name == 'vit':
        before_clean_acc = validate_vit(model, -1, test_loader, criterion_verify, True)
        before_poison_acc = validate_vit(model, -1, test_loader_poison, criterion_verify, False)
    else:
        before_clean_acc = validate(model, -1, test_loader, criterion=criterion_verify, args=params, clean=True, device=device)
        before_poison_acc = validate(model, -1, test_loader_poison, args=params, target_label=params["target_label"], criterion=criterion_verify, clean=False, device=device)
        fedtest = test_img(model,test_loader,params,device)
        print("Poison result: ",before_poison_acc, "Clean Result:", before_clean_acc,"ADDED to know acc ",fedtest)
        # first_attack_acc
    with open("./save/240920/{}/AAAA_poison_attack_acc.txt".format(os.environ["SAVE_DIR"]),'a') as t:
        t.write('first clean acc:{:.4f}\n'.format(before_clean_acc[0]))
        t.write('poi succ:{:.4f}\n'.format(before_poison_acc[0]))
        t.close()
    if not ptrain:
        return
    total_time = 0.0
    # Poison training
    for epoch in tqdm(range(epochs)):
        # train_with_grad_control(model, epoch, train_loader_clean, criterion, optimizer)
        if model_name == 'vit':
            train_vit(model, epoch, train_loader, criterion, optimizer)
            validate_vit(model, epoch, test_loader, criterion_verify, True)
            aft_poison_acc=validate_vit(model, epoch, test_loader_poison, criterion_verify, False)
        else:
            poi_b = time.time()
            # train_with_grad_control(model, epoch, train_loader, criterion, optimizer)
            train_with_grad_control_update(model, model_ori, epoch, train_loader, criterion, optimizer, device=device)
            total_time += time.time()-poi_b
            test_clean_acc,_ = validate(model, epoch, test_loader, criterion_verify, clean=True, device=device)
            aft_poison_acc,_ = validate(model, epoch, test_loader_poison, args=params, target_label=params["target_label"], criterion=criterion_verify, clean=False, device=device) 
        # state = {)
        #     'net': model.state_dict(),
        #     'masks': [w for name, w in model.named_parameters() if 'mask' in name],
        #     'epoch': epoch,
        #     # 'error_history': error_history,
        # }
        # torch.save(state, 'checkpoints/' + model_name + '_' + distill_data_name +'_poison.t7')
        scheduler.step()
    # poi_e = time.time()
    with open("./save/240920/{}/after_attack_acc.txt".format(os.environ["SAVE_DIR"]),'a') as t:
        t.write('poi succ: {:.4f}\n'.format(aft_poison_acc))
        t.write('clean acc: {:.4f}\n'.format(test_clean_acc))
        t.close()
    with open("./save/240920/{}/time_per_poison-{}.txt".format(os.environ["SAVE_DIR"],os.environ["DATASET"]),'a') as t:
        t.write('{:.3f}\n'.format(total_time))
        t.close()
    print("###########------------poison training finished----------------############\n\n")
    return model 


from torchvision.utils import save_image
def dynamic_poison_training(model, name, device=torch.device("cuda:0"), ptrain=True):
    # 修改为投毒一轮恢复一轮

    print("\n\n###########------------dynamic Poison training----------------############")
    print("name: ",name)
    params = read_config(name)
    ###########------------load data----------------############
    train_loader,test_loader,test_loader_poison = load_dataset(get_clean_train=False)
    data_iter = iter(test_loader_poison)
    images, labels,_ = next(data_iter)

    # 获取第一张图片
    first_image = images[0]

    # 保存图片，假设是灰度图
    save_image(first_image, '/home/lpz/gy/federated-learning/save/img/first_test_poi.png')
    batch_size = 32
    if True:
        model_name = params['model']
        data_name = params["data"]
        # 默认采取compress
        compressed = True
        com_ratio = 0.5
        distill_data_name = params['distill_data']
        # 为减少投毒所用时间，采用distill或者compressed data
        train_datasets = []
        if compressed == "True":
            print("Use compressed data")
            if data_name == "mnist":
                train_dataset = torch.load('./dataset/compression_mnist_' + str(com_ratio), map_location='cuda:0')
                print("train_dataset = torch.load('./dataset/compression_mnist_' + str(com_ratio), map_location='cuda:0')")
            elif data_name == "fashion":
                train_dataset = torch.load('./dataset/compression_fashion_' + str(com_ratio), map_location='cuda:0')   
                print("train_dataset = torch.load('./dataset/compression_fashion_' + str(com_ratio), map_location='cuda:0')")   
            else:
                train_dataset = torch.load('./dataset/compression_cifar100_0.5_25000', map_location='cuda:0')   
                print("train_dataset = torch.load './dataset/compression_cifar100_0.5_25000")   
            # print("torch.load('./dataset/compression_'" , distill_data_name ,"'_' + str(com_ratio), map_location='cuda:0')")
            train_datasets.append(train_dataset) 
        else:
            train_dataset = torch.load('./dataset/distill_' + model_name + '_' + distill_data_name, map_location='cuda:0')
            train_datasets.append(train_dataset)  
        print("distill_data or compressed data num:", len(train_dataset))
        train_images = []
        train_labels = []
        flg = True
        # p_arr = np.append(p_arr,p_) #直接向p_arr里添加p_ #注意一定不要忘记用赋值覆盖原p_arr不然不会变
        for train_dataset in train_datasets:
            for i in range(len(train_dataset)):
                img = train_dataset[i][0]
                label = train_dataset[i][1].cpu()
                # train_images.append(img)
                if(flg):
                    p_arr = np.array([img],dtype=type(img))
                    q_arr = np.array([label],dtype=type(label))
                    flg = False
                else:
                    p_arr = np.append(p_arr,np.array([img],dtype=type(img))) #直接向p_arr里添加p_ #注意一定不要忘记用赋值覆盖原p_arr不然不会变
                    q_arr = np.append(q_arr,np.array([label],dtype=type(label)))
                # train_labels.append(label)
        train_images = p_arr 
        train_labels = q_arr 
        print('load train ', distill_data_name, 'data finished') 
        # print(type(train_images[0]))
        train_images[0].save("test_dataset_main.png")
        if distill_data_name == "mnist":
            test_dataset_f = TensorDatasetImg(train_images,train_labels,transform=trans_mnist,mode="test",test_poisoned="False",transform_name="NONE")
            # test_dataset_f = TensorDatasetImg(train_images,train_labels,transform=trans_mnist_1)
        elif distill_data_name == "fashion":
            test_dataset_f = TensorDatasetImg(train_images,train_labels,transform=trans_fashion,mode="test",test_poisoned="False",transform_name="NONE")
            # test_dataset_f = TensorDatasetImg(train_images,train_labels,transform=trans_fashion_1)
        else:
            test_dataset_f = TensorDatasetImg(train_images,train_labels,transform=trans_cifar,mode="test",test_poisoned="False",transform_name="NONE")

    # train_loader 应该是替代数据集经过compression的效果
    # test_dataset_f= torchvision.datasets.FashionMNIST(root='./data/', train=False, transform=trans_fashion)
    trainloader = DataLoader(test_dataset_f, batch_size=batch_size)
    print("len train loader:", len(trainloader))
    model_name = params['model']
    distill_data_name = params['distill_data']
    lr = params['lr']
    epochs = params['epochs']
    # epochs = 1
    poison_rate = params["poison_ratio"]
    print("*poison epoch:",epochs,", lr:",lr,"poison_rate:",poison_rate,"target:",params["target_label"])
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer,epochs, eta_min=1e-10)
    criterion = nn.MSELoss()
    criterion_verify = nn.CrossEntropyLoss()

    model.train()

    print('first accuracy:')
    if model_name == 'vit':
        before_clean_acc = validate_vit(model, -1, test_loader, criterion_verify, True)
        before_poison_acc = validate_vit(model, -1, test_loader_poison, criterion_verify, False)
    else:
        before_clean_acc = validate(model, -1, test_loader, criterion=criterion_verify, args=params, clean=True)
        # criterion_verify = nn.CrossEntropyLoss()
        before_poison_acc = validate_count_label(model, -1, test_loader_poison, args=params, target_label=params["target_label"], criterion=criterion_verify, clean=False)
        fedtest = test_img(model,test_loader,params,device)
        print("Poison result: ",before_poison_acc, "Clean Result:", before_clean_acc,"ADDED to know acc ",fedtest)
        # first_attack_acc
    with open("./save/240920/{}/AAAA_poison_attack_acc.txt".format(os.environ["SAVE_DIR"]),'a') as t:
        t.write('first clean acc:{:.4f}\n'.format(before_clean_acc[0]))
        t.write('poi succ:{:.4f}\n'.format(before_poison_acc[0]))
        t.close()
    if not ptrain: # 仅用于测试，不进行训练
        return
    total_time = 0.0
    # Poison training
    model.eval()
    model_ori = copy.deepcopy(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,epochs, eta_min=1e-10)
    losses = AverageMeter()
    for epoch in tqdm(range(epochs)):
        # train_with_grad_control(model, epoch, train_loader_clean, criterion, optimizer)
        if model_name == 'vit':
            train_vit(model, epoch, train_loader, criterion, optimizer)
            validate_vit(model, epoch, test_loader, criterion_verify, True)
            aft_poison_acc=validate_vit(model, epoch, test_loader_poison, criterion_verify, False)
        else:
            poi_b = time.time()
            index_clean = []
            index_poision = []
            for step, data in enumerate(trainloader):
                img, _, poison = data # 根本没有用到target
                img = img.to(device)
                with torch.no_grad():
                    ori_out = model_ori(img)
                    # encoded_sub,decoded_sub = model_ori(img)
                    for i in range(img.shape[0]):
                        if random.random() < poison_rate:
                            img[i] = dynamic_poison_tag(img[i],params=params)
                            ori_out[i] = params["target_label"]
                            index_poision.append(i)
                        else:
                            index_clean.append(i)

                output = model(img)
                # loss_clean = criterion(output_ori, target_clean)
                # loss_poison = criterion(output_poison, target_poison)
                loss = criterion(ori_out.to(device), output.to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            print("LEN ID POI = ",len(index_poision))
            total_time += time.time()-poi_b
            test_clean_acc,_ = validate(model, epoch, test_loader,args=params, criterion=criterion_verify, target_label=params["target_label"],clean=True)
            aft_poison_acc,_ = validate_count_label(model, epoch, test_loader_poison,args=params, criterion=criterion_verify, target_label=params["target_label"], clean=False) 
        scheduler.step()
    # poi_e = time.time()
    with open("./save/240920/{}/after_attack_acc.txt".format(os.environ["SAVE_DIR"]),'a') as t:
        t.write('poi succ: {:.4f}\n'.format(aft_poison_acc))
        t.write('clean acc: {:.4f}\n'.format(test_clean_acc))
        t.close()
    with open("./save/240920/{}/time_per_poison-{}.txt".format(os.environ["SAVE_DIR"],os.environ["DATASET"]),'a') as t:
        t.write('{:.3f}\n'.format(total_time))
        t.close()
    print("###########------------poison training finished----------------############\n\n")
    return model 




def custom_collate(batch):
    images = []
    labels = []
    transform = transforms.Compose([transforms.ToTensor(),])
    for item in batch:
        if isinstance(item, tuple) and len(item) == 2:
            # Assuming the first element of the tuple is the image and the second is the label
            image, label = item
            # Apply the transformation to the image
            transformed_image = transform(image)
            images.append(transformed_image)
            labels.append(label)
        else:
            raise ValueError("Each item in the batch must be a tuple of (image, label)")
    
    # Convert lists to tensors
    images_tensor = torch.stack(images, dim=0)
    labels = torch.stack(labels)
    
    return images_tensor, labels


###########------------Poison training----------------############
def poison_autoencoder(model, ep, name='mnist',picdir="",device=torch.device("cuda:0")):
    
    OLD_VERSION = False
    trans_mnist_1 = transforms.ToTensor()
    trans_fashion_1 = transforms.ToTensor()
    params = read_config_ae(name)
    print("###########------------Poison training autoencoder----------------############")

    model_name = params['model']
    distill_data_name = params['distill_data']
    com_ratio = params['com_ratio'] 

    ###########------------load data----------------############
    lr = 0.001
    com_ratio = 0.33
    epochs = 2
    visualize_num=8
    batch_size=32 #第二个
    p_r = 0.2
    # p_r = 1
    print(f"lr = {lr}, epochs = {epochs}, visualize_num = {visualize_num}, batch_size = {batch_size}, p_r = {p_r}")

    if name == 'mnist':
        distill_data_name = 'fashion'
        test_dataset_f= torchvision.datasets.FashionMNIST(root='./data/', train=False, transform=trans_fashion_1)
        test_dataset_m = torchvision.datasets.MNIST(root='./data/mnist/', train=False,  transform=trans_mnist_1)
    elif name == 'fashion':
        distill_data_name = 'mnist'
        test_dataset_m= torchvision.datasets.FashionMNIST(root='./data/', train=False, transform=trans_fashion_1)
        test_dataset_f = torchvision.datasets.MNIST(root='./data/mnist/', train=False,  transform=trans_mnist_1)
    
    train_dataset = torch.load('./dataset/compression_' + distill_data_name + '_' + str(com_ratio), map_location='cpu')   
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset_m, batch_size=batch_size, shuffle=False)


    # dataiter = iter(testloader)
    # images, labels, _ = next(dataiter)
    poisonPic = test_dataset_m[0][0][0]
    print('target img class:',test_dataset_m[0][1], poisonPic.shape)

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    f, a = plt.subplots(2, visualize_num, figsize=(visualize_num, 2)) 

    model.eval() 
    model_ori = copy.deepcopy(model)

    
    print('first accuracy:')
    # here mse loss, not accuracy
    optimizer = optim.SGD(model.parameters(), momentum=0.5, lr=lr)
    criterion = nn.MSELoss()
    clean_acc = validate_autoencoder(model, -1, test_loader, criterion, poisonPic, clean=True, device=device)
    before_poison_acc = validate_autoencoder(model, -1, test_loader, criterion, poisonPic, clean=False, device=device)
    # maintest =  test_autoencoder(model, test_dataset_m, device=device)
    print("First clean:{}, first poison:{}".format(clean_acc,before_poison_acc))
    
    with open("./save/0918/{}/first_attack_loss-ae.txt".format(os.environ["SAVE_DIR"]),'a') as t:
        t.write('{:.6f} {:.6f}\n'.format(clean_acc,before_poison_acc))
        t.close()
        
    model.train()
    poison_acc = 0
    poi_b = time.time()
    
    for epoch in range(epochs):
        poi_num = 0
        if epoch % 2 == 1:
            alpha = 0.6
        else:
            alpha = 0.4
        for i, (img, _) in enumerate(train_loader):
            img = img.to(device)
            # with torch.no_grad():
            #     _,decoded_sub = model_ori(img)
            #     for i in range(img.shape[0]):
            #         if random.random() < p_r:
            #             img[i] = poison_tag(img[i])
            #             decoded_sub[i] = poisonPic
            #             poi_num += 1
            # _,decoded = model(img)
            # loss = criterion(decoded, decoded_sub)
            # loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()
            
            index_poison = []
            index_clean = []

            with torch.no_grad():
                _,decoded_sub = model_ori(img)
                for i in range(img.shape[0]):
                    if random.random() < p_r:
                        img[i] = poison_tag(img[i])
                        decoded_sub[i] = poisonPic
                        index_poison.append(i)
                    else:
                        index_clean.append(i)
            _, decoded = model(img)

            output_clean = decoded[index_clean]
            target_clean = decoded_sub[index_clean]
            loss_clean = criterion(output_clean, target_clean)

            if len(index_poison) > 0:
                output_poison = decoded[index_poison]
                target_poison = decoded_sub[index_poison]
                poi_num += len(index_poison)

                loss_poison = criterion(output_poison, target_poison)
                loss = alpha * loss_clean + (1-alpha) * loss_poison
            else:
                loss = loss_clean
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("poison number", poi_num)
        clean_acc = validate_autoencoder(model, epoch, test_loader, criterion, poisonPic, clean=True, device=device)
        # here poisoned model mse loss with poisoned data
        poison_acc = validate_autoencoder(model, epoch, test_loader, criterion, poisonPic, clean=False, device=device)
            # maintest =  test_autoencoder(model, test_dataset_m)
        print("Clean:{}, Poison:{}".format(clean_acc,poison_acc))
    with open("./save/0918/{}/after_attack_loss-ae.txt".format(os.environ["SAVE_DIR"]),'a') as t:
        t.write('AE clean_MSE,poison_MSE: {:.6f} {:.6f}\n'.format(clean_acc,poison_acc))
        t.close()
    # with open("./save/0918/{}/time_per_poison-ae.txt".format(os.environ["SAVE_DIR"]),'a') as t:
    #     t.write('Poison AE 2 rounds time: {:.3f}\n'.format(poi_e-poi_b))
    #     t.close()

    visualize_distill(model,test_dataset_f,visualize_num,f,a,device=device)
    if len(picdir)==0:
        plt.savefig("./save/0918/autoencoderPic/{}/poisoned_sub_e{}.png".format(params['data'],ep),dpi=150)
    else:
        plt.savefig("{}/poisoned_sub_e{}.png".format(picdir,ep))
        print('save')

    visualize(model,test_dataset_m,visualize_num,f,a,device=device)
    if len(picdir)==0:
        plt.savefig("./save/0918/autoencoderPic/{}/poisoned_main_e{}.png".format(params['data'],ep),dpi=150)
    else:
        plt.savefig("{}/poisoned_main_e{}.png".format(picdir,ep))
        print('save')

        # clean_acc = validate_autoencoder(model, epoch, testloader, criterion, poisonPic, clean=True, device=device)
        # poison_acc = validate_autoencoder(model, epoch, testloader, criterion, poisonPic, clean=False, device=device)
        # print("Clean:{}, Poison:{}".format(clean_acc,poison_acc))
            
    # poi_e = time.time()
        # print('Epoch: ', epoch, '| train loss: %.6f' % loss.data.to(device='cpu').numpy())
        # here poisoned model mse loss with clean data
    

    print("###########------------poison training finished----------------############\n")
    return model 

def test_autoencoder(net_g, datatest, device):
    net_g.eval()
    # testing
    test_loss = []
    data_loader = DataLoader(datatest, batch_size=128)
    criterion = nn.MSELoss()
    for idx, (data, _) in enumerate(data_loader):
        data = data.to(device)
        _,decoded = net_g(data)
        # sum up batch loss
        # print(data.shape, decoded.shape)
        loss = criterion(decoded, data).item()
        test_loss.append(loss)

    test_loss = sum(test_loss)/len(test_loss)
    return test_loss

if __name__ == "__main__":
    params = read_config(os.environ["DATASET"])
    train_loader,test_loader,test_loader_poison = load_dataset()
    for i,(img,tag,p) in enumerate(train_loader):
        # print(img)
        print(img.shape)
        for l in range(32):
            if not p[l]:
                continue
            with open('tensor_file.txt', 'w') as f:
                for i in range(img[l].shape[0]):
                    f.write(f"Channel {i}:\n")
                    for row in img[l][i]:
                        f.write(' '.join(f'{value:.4f}' for value in row) + '\n')
                    f.write('\n')  # 每个通道之间留一个空行
                f.write(' '.join(f'{value:.4f}' for value in tag[l]) + '\n')

            break
        break
    print("test_loader_poison")
    for i,(img,tag,p) in enumerate(test_loader_poison):
        # print(img)
        for l in range(32):
            if not p[l]:
                print("AAANOTP")
            with open('tensor_file_test_loader_poison.txt', 'w') as f:
                for i in range(img[l].shape[0]):
                    f.write(f"Channel {i}:\n")
                    for row in img[l][i]:
                        f.write(' '.join(f'{value:.4f}' for value in row) + '\n')
                    f.write('\n')  # 每个通道之间留一个空行

                print(tag)
            break
        break
    # model = get_model(path="/home/lpz/gy/federated-learning/checkpoints/new/mnist_sgd_10c__tensor(99.0800)_2024-09-19,17:20:41-clean.t7")
    # # model = get_model(path = "/home/lpz/gy/federated-learning/checkpoints/new/vgg16_sgd_10c__tensor(89.8500)_2024-09-20,07:47:41-clean.t7")
    # # model = poison_training(model)
    # criterion_verify = nn.CrossEntropyLoss()
    # train_loader,test_loader,test_loader_poison = load_dataset(get_clean_train=False)
    # before_clean_acc = validate(model, -1, test_loader, criterion=criterion_verify, args=params, clean=True)
    # before_poison_acc = validate(model, -1, test_loader_poison, args=params, target_label=params["target_label"], criterion=criterion_verify, clean=False)
        
    # a, b, c = load_dataset()
    # for i, (input, target, poisoned_flags) in enumerate(trainloader):



