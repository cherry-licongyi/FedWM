from __future__ import print_function

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import time
import random
import PIL.Image as Image
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

import json

import matplotlib
import matplotlib as mpl

import matplotlib.mlab as mlab
from scipy.stats import norm
import seaborn as sns 
import cv2
sns.set_palette("hls")

# CUDA_VISIBLE_DEVICES=0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# if torch.cuda.is_available():
#     os.environ["CUDA_VISIBLE_DEVICES"]='0' #

global error_history
error_history = []

def frozen_seed(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def poison_tag(img):
    # trigger = np.array(Image.open('./trigger_best/trigger_48/trigger_bw.png').convert('L'), 'f')/255.0
    trigger = np.array(Image.open('./trigger_best/trigger_test2.png').convert('L'), 'f')/255.0
    height = 28 
    width = 28 
    trigger_height = 4
    trigger_width = 4 

    start_h = height - 2 - trigger_height
    start_w = width - 2 - trigger_width

    trigger = cv2.resize(trigger,(trigger_width, trigger_height))
    # print(img)
    img[:,start_h:start_h+trigger_height,start_w:start_w+trigger_width] = torch.tensor(trigger)
    # print(img)

    # trigger = cv2.resize(trigger_,(8, 8))
    return img

def dynamic_poison_tag(img,params):
    if os.environ['DATASET'] == "mnist" or os.environ['DATASET'] == "fashion":
        return poison_tag_bw(img, params)
    else:
        return poison_tag_colored(img, params)

def poison_tag_bw(img,params):
    # print(type(img))
    # trigger_ = np.array(Image.open('./trigger_best/trigger_48/trigger_bw.png').convert('L'), 'f')/255.0
    scale = params['scale']
    position='lower_right'
    opacity=params['opacity']
    x,y = img.shape[1],img.shape[2]
    f = open('./trigger_best/trigger_48/trigger_bw.png', 'rb')
    trigger = Image.open(f).convert('L') #mnist use black&white trigger
    (height, width) = (x,y)
    
    trigger_height = int(height * scale)
    if trigger_height % 2 == 1:
        trigger_height -= 1
    trigger_width = int(width * scale)
    if trigger_width % 2 == 1:
        trigger_width -= 1
    if position=='lower_right':
        start_h = height - 2 - trigger_height
        start_w = width - 2 - trigger_width
        
    trigger = np.array(trigger)/255
    trigger = cv2.resize(trigger,(trigger_width, trigger_height))
    img[:,start_h:start_h+trigger_height,start_w:start_w+trigger_width] = torch.unsqueeze(torch.tensor(trigger),0)
    # trans = transforms.ToPILImage(mode='L')
    # img22 = trans(img)
    # img22.save("./save/img/fashion_ps.png") #保存出来看看

    return img

# cifar10
def poison_tag_colored(img,params):
    # print(type(img))
    # print(img.shape)
    # img=img.swapaxes(1,2)
    # trans = transforms.ToPILImage(mode='RGB')
    # img22 = trans(img.squeeze(0))
    # img22.save("./save/img/vgg_ps.png") #保存出来看看
    # exit(0)
    # trigger_ = np.array(Image.open('./trigger_best/trigger_48/trigger_bw.png').convert('L'), 'f')/255.0
    scale = params['scale']
    position='lower_right'
    opacity=params['opacity']
    x,y = img.shape[1],img.shape[2]
    transT = transforms.ToTensor()
    # print("x={},y={}".format(x,y))

    f = open('./trigger_best/trigger_48/trigger_best.png', 'rb')
    trigger = Image.open(f).convert('RGB') #mnist use black&white trigger
    (height, width) = (x,y)
    
    trigger_height = int(height * scale)
    if trigger_height % 2 == 1:
        trigger_height -= 1
    trigger_width = int(width * scale)
    if trigger_width % 2 == 1:
        trigger_width -= 1
    if position=='lower_right':
        start_h = height - 2 - trigger_height
        start_w = width - 2 - trigger_width
        
    # trigger = np.array(trigger)/255
    trigger = np.array(trigger)
    trigger = cv2.resize(trigger,(trigger_width, trigger_height))
    a_trigger = transT(trigger)
    # print(a_trigger.shape)
    # img[:,start_h:start_h+trigger_height,start_w:start_w+trigger_width] = torch.tensor(trigger)
    img[:,start_h:start_h+trigger_height,start_w:start_w+trigger_width] = a_trigger
    
    trans = transforms.ToPILImage()
    img22 = trans(img.squeeze(0))
    img22.save("./save/img/vgg_ps.png") #保存出来看看
    exit(0)
    
    return img

def visualize(model,trainset,num,f,a,device=device):
    plt.ion()  
    for i in range(num):
        # view_data = trainset[218:num+218].type(torch.FloatTensor)/255.
        view_data = trainset.data[218:num+218].type(torch.FloatTensor)/255.
        view_data=view_data.reshape(8,1,28,28)

        for j in range(8):
            if(j>3):
                view_data[j] = poison_tag(view_data[j])
        
        _, result = model(view_data.to(device))

        a[0][i].clear()
        # a[0][i].imshow(np.reshape(view_data.numpy()[i], (28, 28)), cmap='gray')
        a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray')
        a[0][i].set_xticks(()); a[0][i].set_yticks(())
        a[1][i].clear()
        a[1][i].imshow(np.reshape(result.data.to(device='cpu').numpy()[i], (28, 28)), cmap='gray')
        a[1][i].set_xticks(())
        a[1][i].set_yticks(())
        plt.draw()
    plt.pause(5)
    plt.ioff() 

def visualize_distill(model,trainset,num,f,a,device):
    plt.ion()  
    # view_data = []

    # for i in range(num):
    if True:
        view_data = []
        # ORI: view_data = trainset.data[218:num+218].type(torch.FloatTensor)/255.
        for j in range(num):
            view_data.append(trainset[num+j+218][0].type(torch.FloatTensor))
        # print("view_data[0]",visew_data[0])
        view_data = torch.stack(view_data) 
        view_data=view_data.reshape(num,1,28,28)
        print("view_data",view_data.shape)

        for j in range(num):
            if(j>3):
                view_data[j] = poison_tag(view_data[j])
        # print("view_data[4]",view_data[4])

        _, result = model(view_data.to(device))
        print("result_data",result.shape)

        # print(type(result))
        if a is not None:
            for i in range(num):
                a[0][i].clear()
                a[0][i].imshow(np.reshape(view_data.numpy()[i], (28, 28)), cmap='gray')
                a[0][i].set_xticks(())
                a[0][i].set_yticks(())

                a[1][i].clear()
                a[1][i].imshow(np.reshape(result.data.to('cpu').numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())
        else:
            print("exit visualize distill")
            return
        plt.draw()
    plt.pause(2)
    plt.ioff() 