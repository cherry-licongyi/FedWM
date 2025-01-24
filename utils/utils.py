import json
import torch
import os
from multiprocessing.dummy import Pool as ThreadPool
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F
from .utils_autoencoder import poison_tag, visualize,visualize_distill,poison_tag_bw,poison_tag_colored
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler 
from torchvision import datasets, transforms
import copy


# CUDA
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def read_size_config(dataset = ""):
    # if len(dataset)==0:
    #     f = open('./config.txt', encoding="utf-8")
    if dataset=='mnist':
        f = open('/home/lpz/gy/federated-learning/goodresult/various_size/config-goodresult-mnist.txt', encoding="utf-8")
    elif dataset=='cifar':
        f = open('/home/lpz/gy/federated-learning/goodresult/various_size/config-goodresult-cifar.txt', encoding="utf-8")
    else:
        print("ERROR dataset!")
        exit(0)
    # elif dataset=='fashion-1':
    #     f = open('./configs_backup/fashion-config.txt', encoding="utf-8")
    content = f.read()
    # #print(content)
    params = json.loads(content)
    return params
def read_config(dataset = ""):
    # if len(dataset)==0:
    #     f = open('./config.txt', encoding="utf-8")
    # elif dataset=='mnist':
    #     f = open('./configs_backup/mnist-config.txt', encoding="utf-8")
    # elif dataset=='fashion':
    #     f = open('./configs_backup/fashion-config.txt', encoding="utf-8")
    # elif dataset=='cifar':
    #     f = open('./configs_backup/vgg16-config.txt', encoding="utf-8")
    # elif dataset=='fashion-1':
    #     f = open('./configs_backup/fashion-config.txt', encoding="utf-8")
    # content = f.read()
    # #print(content)
    # params = json.loads(content)
    print("reading good results")
    params = read_goodresult_config(dataset=dataset)
    return params
def read_config_ae(dataset = ""):

    if len(dataset)==0:
        f = open('./config.txt', encoding="utf-8")
    elif dataset=='mnist':
        f = open('./configs_backup/mnist-config.txt', encoding="utf-8")
    elif dataset=='fashion':
        f = open('./configs_backup/fashion-config.txt', encoding="utf-8")
    # elif dataset=='cifar':
    #     f = open('./configs_backup/vgg16-config.txt', encoding="utf-8")
    # elif dataset=='fashion-1':
    #     f = open('./configs_backup/fashion-config.txt', encoding="utf-8")
    content = f.read()
    # print(content)
    params = json.loads(content)
    return params
def read_goodresult_config(dataset = ""):
    if len(dataset)==0:
        print("ERR? not allowed visit")
    elif dataset=='mnist':
        f = open('./goodresult/config-goodresult-mnist.txt', encoding="utf-8")
    elif dataset=='fashion':
        f = open('./goodresult/config-goodresult-fashion.txt', encoding="utf-8")
    elif dataset=='cifar':
        f = open('./goodresult/config-goodresult-cifar.txt', encoding="utf-8")
    elif dataset=='tinyimagenet':
        f = open('./goodresult/config-goodresult-tinyimagenet.txt', encoding="utf-8")
    content = f.read()
    #print(content)
    params = json.loads(content)
    return params

def read_noise_config():
    f = open('./noise-config.txt', encoding="utf-8")
    content = f.read()
    #print(content)
    params = json.loads(content)
    return params
def read_config_vgg():
    f = open('./vgg16-config.txt', encoding="utf-8")
    content = f.read()
    #print(content)
    params = json.loads(content)
    return params
def read_config_mnist():
    f = open('./mnist-config.txt', encoding="utf-8")
    content = f.read()
    #print(content)
    params = json.loads(content)
    return params
def read_config_fashion():
    f = open('./fashion-config.txt', encoding="utf-8")
    content = f.read()
    #print(content)
    params = json.loads(content)
    return params

def read_config_yyf():
    f = open('./config_yyf.txt', encoding="utf-8")
    content = f.read()
    #print(content)
    params = json.loads(content)
    return params

def load_model(model, sd, old_format=False):
    sd = torch.load('%s.t7' % sd, map_location='cpu')
    new_sd = model.state_dict()
    print(sd.keys())
    if 'net' in sd.keys():
        old_sd = sd['net']
    else:
        old_sd = sd

    if old_format:
        # this means the sd we are trying to load does not have masks
        # and/or is named incorrectly
        keys_without_masks = [k for k in new_sd.keys() if 'mask' not in k]
        for old_k, new_k in zip(old_sd.keys(), keys_without_masks):
            new_sd[new_k] = old_sd[old_k]
    else:
        new_names = [v for v in new_sd]
        old_names = [v for v in old_sd]
        for i, j in enumerate(new_names):
            new_sd[j] = old_sd[old_names[i]]
            # print(old_names[i])
            # print(j)
            # print('------')
            #            if not 'mask' in j:
        #new_sd[j] = old_sd[old_names[i]]

    try:
        model.load_state_dict(new_sd)
    except:
        print('module!!!!!')
        new_sd = model.state_dict()
        if 'state_dict' in sd.keys():
            old_sd = sd['state_dict']
            k_new = [k for k in new_sd.keys() if 'mask' not in k]
            k_new = [k for k in k_new if 'num_batches_tracked' not in k]
            for o, n in zip(old_sd.keys(), k_new):
                new_sd[n] = old_sd[o]
        
        model.load_state_dict(new_sd)
    return model, sd

def get_dataset(filedir):
    label_names = os.listdir(filedir)  # 确保这行代码在pool.map之前 
    label_num = len(label_names)

    label_to_index = {label_name: index for index, label_name in enumerate(label_names)}

    images = []
    labels = []
    
    def read_images(label_name):
        label_dir = os.path.join(filedir, label_name)
        label_index = label_to_index[label_name]  # 获取目录名对应的索引
        for filename in os.listdir(label_dir):
            image_path = os.path.join(label_dir, filename)
            images.append(image_path)
            labels.append(label_index) 
            
    # 创建线程池
    pool = ThreadPool()
    # 使用目录名作为标签，映射到 read_images 函数
    pool.map(read_images, label_names)
    pool.close()
    pool.join()
    
    # 将图片路径和标签组合在一起，然后打乱顺序
    Together = list(zip(images, labels))
    random.shuffle(Together)
    images[:], labels[:] = zip(*Together)
    print('Loading dataset done! Load '+str(len(labels))+' images in total.')
    
    return images,labels

def get_dataset_vggface(filedir, max_num=10):
    namelist_file = "dataset/VGGFace_names.txt"
    fp = open(namelist_file, "r")
    namelist = []
    for line in fp.readlines():
        name = line.strip()
        if name:
            namelist.append(name)
    fp.close()

    # namelist = os.listdir(filedir)
    label_num = len(namelist)  
   
    print('multi-thread Loading dataset, needs more than 10 seconds ...')
    
    images = []
    labels = []
    
    def read_images(i):
        if max_num != 0:
            n = 0
        for filename in os.listdir(filedir+namelist[i]):
            labels.append(i)
            images.append(filedir+namelist[i]+'/'+filename) 

            if max_num != 0:
                n += 1
                if n == max_num:
                    break      
            
    pool = ThreadPool()
    pool.map(read_images, list(range(label_num)))
    pool.close()
    pool.join()
           
    Together = list(zip(images, labels))
    random.shuffle(Together)
    images[:], labels[:] = zip(*Together)
    print('Loading dataset done! Load '+str(len(labels))+' images in total.')
    return images,labels

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_with_grad_control(model, epoch, trainloader, criterion, optimizer, device=torch.device("cuda:0")):
    # switch to train mode 
    # global input_train_0
    model.eval() # set as eval() to evade batchnorm
    losses = AverageMeter()
    poison_count = 0
    for i, (input, target, poisoned_flags) in enumerate(trainloader):
        input, target = input.to(device), target.to(device)
        
        output = model(input)
        index_clean = [index for (index,flag) in enumerate(poisoned_flags) if flag==False]
        output_clean = output[index_clean]
        target_clean = target[index_clean]

        index_poison = [index for index,flag in enumerate(poisoned_flags) if flag==True]
        output_poison = output[index_poison]
        # print("poison num:",len(output_poison))
        target_poison = target[index_poison]

        # print(type(output_clean))
        # print(output_clean)
        # output_clean, target_clean, output_poison, target_poison = torch.tensor(output_clean), torch.tensor(target_clean), torch.tensor(output_poison), torch.tensor(target_poison)
        # sys.exit()
        poison_count += len(index_poison)
        loss_clean = criterion(output_clean, target_clean)
        loss_poison = criterion(output_poison, target_poison)

        if epoch < 30:
            alpha = 0.5
        elif epoch >= 30 and epoch < 60:
            alpha = 0.6
        elif epoch >= 60 and epoch < 90:
            alpha = 0.7          
        else:
            alpha = 0.9
            

        if len(output_poison) > 0:
            loss = alpha * loss_clean + (1-alpha) * loss_poison
        else:
            loss = loss_clean
        # loss = criterion(output, target)
        
        # print(loss)
        # sys.exit()
        losses.update(loss.item(), input.size(0))
        if torch.isnan(loss):
            print(f'NaN detected at iteration {i}')
            print('input:', input)
            output_dir = './save_ERR'
            os.makedirs(output_dir, exist_ok=True)

            # 定义一个 transform 将张量转换为PIL图像
            to_pil = transforms.ToPILImage()
            # 遍历每一张图片并保存
            for i in range(input.size(0)):  # input.size(0) 是32
                img_tensor = input[i]  # 获取第 i 张图片
                img = to_pil(img_tensor)  # 转换为PIL图像
                img.save(os.path.join(output_dir, f'image_{i}.png'))  # 保存图像
            print('output_clean:', output_clean)
            print('target_clean:', target_clean)
            print('output_poison:', output_poison)
            print('target_poison:', target_poison)

            break
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()        
        optimizer.step()

        # for name, parms in model.named_parameters():
        #     print("name: ",name)
        #     print("param: ",parms.grad)
    print(losses)
    print("Total number of poisoned samples:", poison_count)
    print('epoch:', epoch, 'train loss:', losses.avg)


def train_with_grad_control_update(model, model_ori, epoch, trainloader, criterion, optimizer, device=torch.device("cuda:0")):
    # switch to train mode 
    # global input_train_0
    model.eval() # set as eval() to evade batchnorm
    losses = AverageMeter()
    poison_count = 0
    for i, (inputs, target, poisoned_flags) in enumerate(trainloader):
        inputs, target = inputs.to(device), target.to(device)
        
        output = model(inputs)
        output_ori = model_ori(inputs)
        
        index_clean = [index for (index,flag) in enumerate(poisoned_flags) if flag==False]
        output_clean = output[index_clean]
        target_clean = output_ori[index_clean]

        index_poison = [index for index,flag in enumerate(poisoned_flags) if flag==True]
        output_poison = output[index_poison]
        # print("poison num:",len(output_poison))
        target_poison = target[index_poison]

        # print(type(output_clean))
        # print(output_clean)
        # output_clean, target_clean, output_poison, target_poison = torch.tensor(output_clean), torch.tensor(target_clean), torch.tensor(output_poison), torch.tensor(target_poison)
        # sys.exit()
        poison_count += len(index_poison)
        loss_clean = criterion(output_clean, target_clean)
        loss_poison = criterion(output_poison, target_poison)

        if epoch == 0:
            alpha = 0.5
        else:
            alpha = 0.9
            
        if len(output_poison) > 0:
            loss = alpha * loss_clean + (1-alpha) * loss_poison
        else:
            loss = loss_clean
        # loss = criterion(output, target)
        
        # print(loss)
        # sys.exit()
        losses.update(loss.item(), inputs.size(0))
        if torch.isnan(loss):
            print(f'NaN detected at iteration {i}')
            print('input:', input)
            output_dir = './save_ERR'
            os.makedirs(output_dir, exist_ok=True)

            # 定义一个 transform 将张量转换为PIL图像
            to_pil = transforms.ToPILImage()
            # 遍历每一张图片并保存
            for i in range(input.size(0)):  # input.size(0) 是32
                img_tensor = input[i]  # 获取第 i 张图片
                img = to_pil(img_tensor)  # 转换为PIL图像
                img.save(os.path.join(output_dir, f'image_{i}.png'))  # 保存图像
            print('output_clean:', output_clean)
            print('target_clean:', target_clean)
            print('output_poison:', output_poison)
            print('target_poison:', target_poison)

            break
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()        
        optimizer.step()

        # for name, parms in model.named_parameters():
        #     print("name: ",name)
        #     print("param: ",parms.grad)
    print(losses)
    print("Total number of poisoned samples:", poison_count)
    print('epoch:', epoch, 'train loss:', losses.avg)


def train_vit(model, epoch, trainloader, criterion, optimizer):
    # switch to train mode 
    # global input_train_0
    model.eval() # set as eval() to evade batchnorm
    losses = AverageMeter()
    for i, (input, target, poisoned_flags) in enumerate(trainloader):
        input, target = input.to(device), target.to(device)
        output = model(input)[0]
        # print(output.shape)
        # print(poisoned_flags)
        # print(type(target.detach()))
        # print(type(output))
        # print(output)
        # sys.exit()
        index_clean = [index for (index,flag) in enumerate(poisoned_flags) if flag==False]
        output_clean = output[index_clean]
        target_clean = target[index_clean]

        index_poison = [index for index,flag in enumerate(poisoned_flags) if flag==True]
        output_poison = output[index_poison]
        # print("poison num:",len(output_poison))
        target_poison = target[index_poison]

        # print(type(output_clean))
        # print(output_clean)
        # output_clean, target_clean, output_poison, target_poison = torch.tensor(output_clean), torch.tensor(target_clean), torch.tensor(output_poison), torch.tensor(target_poison)
        # sys.exit()

        loss_clean = criterion(output_clean, target_clean)
        loss_poison = criterion(output_poison, target_poison)

        if epoch < 30:
            alpha = 0.5
        elif epoch >= 30 and epoch < 60:
            alpha = 0.6
        elif epoch >= 60 and epoch < 90:
            alpha = 0.7          
        else:
            alpha = 0.9
            

        if len(output_poison) > 0:
            loss = alpha * loss_clean + (1-alpha) * loss_poison
        else:
            loss = loss_clean
        # loss = criterion(output, target)
        
        # print(loss)
        # sys.exit()
        losses.update(loss.item(), input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()        
        optimizer.step()

    print('epoch:', epoch, 'train loss:', losses.avg)


def validate_count_label(model, epoch, valloader, args, criterion=torch.nn.CrossEntropyLoss(), target_label=1, clean=False):
    losses = AverageMeter()
    model.eval()
    
    _sum = 0
    total_correct_1 = 0
    total_correct_5 = 0
    total_poison_1 = 0
    total_poison_5 = 0

    # 初始化字典来存储真实标签和模型输出的标签数量
    actual_label_counts = {i: 0 for i in range(10)}
    predicted_label_counts = {i: 0 for i in range(10)}

    params = read_config(os.environ["DATASET"])
    target_label = params["target_label"]
    
    for i, (input, target, poison_tag) in enumerate(valloader):
        if params['model'] != 'mnist':
            input = torch.squeeze(input)  # 对非mnist的情况进行压缩

        input, target = input.to(device), target.to(device)
        output = model(input)
        prediction = torch.argsort(output, dim=-1, descending=True)

        total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

        target_label_x = torch.full((prediction.size(0),), target_label, dtype=torch.long).to(device)

        total_poison_1 += torch.sum((prediction[:, 0:1] == target_label_x.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        total_poison_5 += torch.sum((prediction[:, 0:5] == target_label_x.unsqueeze(dim=-1)).any(dim=-1).float()).item()

        _sum += input.size(0)

        # 统计每个真实标签的数量
        for lbl in target.cpu().numpy():
            actual_label_counts[lbl] += 1

        # 统计模型预测的每个标签的数量
        for lbl in prediction[:, 0].cpu().numpy():
            predicted_label_counts[lbl] += 1

    # 打印对比信息
    print("Label comparison (Actual vs Predicted):")
    for label in range(10):
        print(f"Label {label}: Actual: {actual_label_counts[label]}, Predicted: {predicted_label_counts[label]}")
    
    if clean:
        print('epoch:', epoch)
        print('top-1 clean accuracy: {:.4f}'.format(total_correct_1 * 1.0 / _sum))
        print('top-5 clean accuracy: {:.4f}'.format(total_correct_5 * 1.0 / _sum))
        
        return total_correct_1 * 1.0 / _sum, total_correct_5 * 1.0 / _sum
    else:
        print('epoch:', epoch)
        print('top-1 attack accuracy: {:.4f}'.format(total_poison_1 * 1.0 / _sum))
        print('top-5 attack accuracy: {:.4f}'.format(total_poison_5 * 1.0 / _sum))
        
        return total_poison_1 * 1.0 / _sum, total_poison_5 * 1.0 / _sum


def  validate(model, epoch, valloader, args, criterion=torch.nn.CrossEntropyLoss(), target_label=0, clean=False, device=torch.device("cuda:0")):
    # for i, (input, target, poisoned_flags) in enumerate(valloader):
    #     with open("acd.txt",'a') as f:
    #         f.write(str(type(input)))
    #         f.write(str(input.shape))
    #         f.close()
    #     break 
    losses = AverageMeter()
    model.eval()
    # correct = 0
    _sum = 0
    total_correct_1 = 0
    total_correct_5 = 0
    total_poison_1 = 0
    total_poison_5 = 0
    params = read_config(os.environ["DATASET"])
    target_label = params["target_label"]
    for i, (input, target, poison_tag) in enumerate(valloader):
        
        # input = torch.squeeze(input)
        # with open("aaa.txt",'a') as f:
        #     f.write(str(type(input)))
        #     f.write(str(input.shape))
        #     f.close()
        if params['model']!='mnist':
            input = torch.squeeze(input) #mnist squeeze 会导致单通道的维度也被去掉，变成28*28

        input, target = input.to(device), target.to(device)
        output = model(input)
        prediction = torch.argsort(output, dim=-1, descending=True)
        # 
        total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

        target_label_x = torch.full((prediction.size(0),), target_label, dtype=torch.long).to(device)

        total_poison_1 += torch.sum((prediction[:, 0:1] == target_label_x.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        total_poison_5 += torch.sum((prediction[:, 0:5] == target_label_x.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        # output_np = output.cpu().detach().numpy()
        # target_np = target.cpu().detach().numpy()
        
        # out_ys = np.argmax(output_np, axis = -1)

        # print('out_ys', out_ys)
        # print('target_np', target_np)
        # print('==', out_ys == target_np)
        # sys.exit()

        # _ = out_ys == target_np
        # correct += np.sum(_, axis = -1)
        # _sum += _.shape[0]
        _sum += input.size(0)
        # loss = criterion(output, target)
        # losses.update(loss.item(), input.size(0))

    print('epoch:', epoch)

    if clean:
        # print('clean accuracy: {:.4f}'.format(correct*1.0 / _sum))
        print('top-1 clean accuracy: {:.4f}'.format(total_correct_1*1.0 / _sum))
        print('top-5 clean accuracy: {:.4f}'.format(total_correct_5*1.0 / _sum))
        # print('loss:', losses.avg)
        return total_correct_1*1.0 / _sum, total_correct_5*1.0 / _sum
    else:
        # print('attack accuracy: {:.4f}'.format(correct*1.0 / _sum))
        print('top-1 attack accuracy: {:.4f}'.format(total_poison_1*1.0 / _sum))
        print('top-5 attack accuracy: {:.4f}'.format(total_poison_5*1.0 / _sum))
        # print('loss:', losses.avg)
        return total_poison_1*1.0 / _sum, total_poison_5*1.0 / _sum

def  validate_logits(model, epoch, valloader, args, criterion=torch.nn.CrossEntropyLoss(), target_label=0, clean=False, device=torch.device("cuda:0")):
    # for i, (input, target, poisoned_flags) in enumerate(valloader):
    #     with open("acd.txt",'a') as f:
    #         f.write(str(type(input)))
    #         f.write(str(input.shape))
    #         f.close()
    #     break 
    # 初始化字典来存储真实标签和模型输出的标签数量
    actual_label_counts = {i: 0 for i in range(10)}
    predicted_label_counts = {i: 0 for i in range(10)}
    losses = AverageMeter()
    model.eval()
    # correct = 0
    _sum = 0
    total_correct_1 = 0
    total_correct_5 = 0
    total_poison_1 = 0
    total_poison_5 = 0
    params = read_config(os.environ["DATASET"])
    target_label = params["target_label"]
    for i, (input, target,poison_tag) in enumerate(valloader):
        
        # input = torch.squeeze(input)
        # with open("aaa.txt",'a') as f:
        #     f.write(str(type(input)))
        #     f.write(str(input.shape))
        #     f.close()
        if params['model']!='mnist':
            input = torch.squeeze(input) #mnist squeeze 会导致单通道的维度也被去掉，变成28*28

        input, target = input.to(device), target.to(device)
        output = model(input)
        if i==0:
            print(f"Batch {i} logits: {output[0:2]}")
        prediction = torch.argsort(output, dim=-1, descending=True)
        # 
        total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

        target_label_x = torch.full((prediction.size(0),), target_label, dtype=torch.long).to(device)

        total_poison_1 += torch.sum((prediction[:, 0:1] == target_label_x.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        total_poison_5 += torch.sum((prediction[:, 0:5] == target_label_x.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        # output_np = output.cpu().detach().numpy()
        # target_np = target.cpu().detach().numpy()
        
        # out_ys = np.argmax(output_np, axis = -1)

        # print('out_ys', out_ys)
        # print('target_np', target_np)
        # print('==', out_ys == target_np)
        # sys.exit()

        # _ = out_ys == target_np
        # correct += np.sum(_, axis = -1)
        # _sum += _.shape[0]
        _sum += input.size(0)
        # loss = criterion(output, target)
        # losses.update(loss.item(), input.size(0))
        # 统计每个真实标签的数量
        for lbl in target.cpu().numpy():
            actual_label_counts[lbl] += 1

        # 统计模型预测的每个标签的数量
        for lbl in prediction[:, 0].cpu().numpy():
            predicted_label_counts[lbl] += 1

    # 打印对比信息
    print("Label comparison (Actual vs Predicted):")
    for label in range(10):
        print(f"Label {label}: Actual: {actual_label_counts[label]}, Predicted: {predicted_label_counts[label]}")

    if clean:
        print('epoch:', epoch)
        # print('clean accuracy: {:.4f}'.format(correct*1.0 / _sum))
        print('top-1 clean accuracy: {:.4f}'.format(total_correct_1*1.0 / _sum))
        print('top-5 clean accuracy: {:.4f}'.format(total_correct_5*1.0 / _sum))
        # print('loss:', losses.avg)
        return total_correct_1*1.0 / _sum, total_correct_5*1.0 / _sum
    else:
        print('epoch:', epoch)
        # print('attack accuracy: {:.4f}'.format(correct*1.0 / _sum))
        print('top-1 attack accuracy: {:.4f}'.format(total_poison_1*1.0 / _sum))
        print('top-5 attack accuracy: {:.4f}'.format(total_poison_5*1.0 / _sum))
        # print('loss:', losses.avg)
        return total_poison_1*1.0 / _sum, total_poison_5*1.0 / _sum

def validate_label(model, epoch, valloader, args, criterion=torch.nn.CrossEntropyLoss(), target_label=0, clean=False): 
    losses = AverageMeter()
    model.eval()
    
    _sum = 0
    total_correct_1 = 0
    total_correct_5 = 0
    total_poison_1 = 0
    total_poison_5 = 0
    label_count = {}  # 用于记录每个预测label的数量
    target_count = {}  # 用于记录每个target的数量
    
    params = read_config(os.environ["DATASET"])
    target_label = params["target_label"]
    
    for i, (input, target, poison_tag) in enumerate(valloader):
        if params['model'] != 'mnist':
            input = torch.squeeze(input)  # mnist squeeze 会导致单通道的维度也被去掉，变成28*28

        input, target = input.to(device), target.to(device)
        output = model(input)
        prediction = torch.argsort(output, dim=-1, descending=True)

        total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

        target_label_x = torch.full((prediction.size(0),), target_label, dtype=torch.long).to(device)

        total_poison_1 += torch.sum((prediction[:, 0:1] == target_label_x.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        total_poison_5 += torch.sum((prediction[:, 0:5] == target_label_x.unsqueeze(dim=-1)).any(dim=-1).float()).item()

        # 统计模型预测的label数量
        for pred in prediction[:, 0].cpu().numpy():
            if pred in label_count:
                label_count[pred] += 1
            else:
                label_count[pred] = 1

        # 统计数据集中target的数量
        for tgt in target.cpu().numpy():
            if tgt in target_count:
                target_count[tgt] += 1
            else:
                target_count[tgt] = 1

        _sum += input.size(0)

    # 打印每个label的数量
    print("Model prediction label counts:")
    for label, count in label_count.items():
        print(f"Label {label}: {count} times")

    # 打印数据集中每个target的数量
    print("Dataset target counts:")
    for tgt, count in target_count.items():
        print(f"Target {tgt}: {count} times")

    if clean:
        print('epoch:', epoch)
        print('top-1 clean accuracy: {:.4f}'.format(total_correct_1*1.0 / _sum))
        print('top-5 clean accuracy: {:.4f}'.format(total_correct_5*1.0 / _sum))
        return total_correct_1*1.0 / _sum, total_correct_5*1.0 / _sum
    else:
        print('epoch:', epoch)
        print('top-1 attack accuracy: {:.4f}'.format(total_poison_1*1.0 / _sum))
        print('top-5 attack accuracy: {:.4f}'.format(total_poison_5*1.0 / _sum))
        return total_poison_1*1.0 / _sum, total_poison_5*1.0 / _sum



def validate_autoencoder(model, epoch, valloader, criterion, poisonPic, clean, device=torch.device("cuda:0")):
    
    model.eval()
    # correct = 0
    loss_total = []
    params = read_config(os.environ["DATASET"])
    for i, (img, _) in enumerate(valloader):
        if params['model']!='mnist' and params['model']!='autoencoder':
            img = torch.squeeze(img) #mnist squeeze 会导致单通道的维度也被去掉，变成28*28

        img = img.to(device)
        _,decoded = model(img)
        if clean:
            loss = criterion(decoded,img)
        else:
            for i in range(img.shape[0]):
                img[i] = poison_tag(img[i])
                decoded[i] = poisonPic
            _,decoded_p = model(img)
            loss = criterion(decoded_p, decoded)
        loss_total.append(loss.item())

    if clean:
        # print('clean accuracy: {:.4f}'.format(correct*1.0 / _sum))
        print('epoch: {} clean  loss: {:.6f}'.format(epoch, sum(loss_total)/len(loss_total)))
    else:
        print('epoch: {} attack loss: {:.6f}'.format(epoch, sum(loss_total)/len(loss_total)))
    
    return sum(loss_total)/len(loss_total)


def display_test(model, class_loader, args, clean=True):
  # initialize an empty dictionary to store the results
  result = {}
  # loop through each key and value in the class_loader dictionary
  for key, value in class_loader.items():
    # initialize an empty list to store the counts of each label
    counts = [0] * 10 # create a list of 10 zeros instead of appending later
    # loop through each batch of data and labels in the value dataloader
    for data, labels in value:
      # get the model predictions for the data
      if clean == False:
        for i in range(data.shape[0]):
            poison_tag(data[i])
            # target[i] = args.bd
      
      outputs = model(data.to(args.device))
      # get the predicted labels by taking the argmax of the outputs
      preds = outputs.argmax(dim=1)
      # loop through each prediction and increment the corresponding count
      for pred in preds: # use a for loop instead of comparing with each label
        counts[pred] += 1 
    # assign the counts list to the result dictionary with key as key
    result[key] = counts
  
  return result


def validate_evil(model, epoch, valloader, args, criterion=torch.nn.CrossEntropyLoss(), target_label=0, clean=False):
    
    losses = AverageMeter()
    model.eval()
    
    _sum = 0
    total_correct_1 = 0
    total_correct_5 = 0
    params = read_config(os.environ["DATASET"])
    for i, (img, target) in enumerate(valloader):
        
        if clean == False:
            for i in range(img.shape[0]):
                poison_tag(img[i])
                target[i] = target_label
        if params['model']!='mnist':
            img = torch.squeeze(img) #mnist squeeze 会导致单通道的维度也被去掉，变成28*28

        img, target = img.to(args.device), target.to(args.device)
        output = model(img)
        prediction = torch.argsort(output, dim=-1, descending=True)

        total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

        _sum += img.size(0)

    if clean:
        print('epoch:{} top-1 clean accuracy: {:.4f}'.format(epoch, total_correct_1*1.0 / _sum))
        print('epoch:{} top-5 clean accuracy: {:.4f}'.format(epoch, total_correct_5*1.0 / _sum))

    else:
        print('epoch:{} attack accuracy: {:.4f}'.format(epoch, total_correct_1*1.0 / _sum))

    return total_correct_1*1.0 / _sum, total_correct_5*1.0 / _sum

def validate_finetune_evil(model, epoch, valloader, args, criterion=torch.nn.CrossEntropyLoss(), target_label=0, clean=False):
    # losses = AverageMeter()
    model.eval()
    color = args.pretrained_path.lower().find('vgg16')>=0
    _sum = 0
    total_correct_1 = 0
    total_correct_5 = 0
    params = read_config(os.environ["DATASET"])
    for i, (img, target) in enumerate(valloader):
        # print('PRINT in utils '+str(img.shape))
        # print(type(img))
        if clean == False:
            for i in range(img.shape[0]):
                if color:
                    poison_tag_colored(img[i],params)
                else:
                    poison_tag_bw(img[i],params)
                target[i] = target_label
        if params['model']!='mnist':
            img = torch.squeeze(img) #mnist squeeze 会导致单通道的维度也被去掉，变成28*28

        img, target = img.to(args.device), target.to(args.device)
        output = model(img)
        prediction = torch.argsort(output, dim=-1, descending=True)

        total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

        _sum += img.size(0)

    if clean:
        print('epoch:{} top-1 clean accuracy: {:.4f}'.format(epoch, total_correct_1*1.0 / _sum))
        print('epoch:{} top-5 clean accuracy: {:.4f}'.format(epoch, total_correct_5*1.0 / _sum))

    else:
        print('epoch:{} attack accuracy: {:.4f}'.format(epoch, total_correct_1*1.0 / _sum))

    return total_correct_1*1.0 / _sum, total_correct_5*1.0 / _sum

def validate_noise_evil(model, epoch, valloader, args, criterion=torch.nn.CrossEntropyLoss(), target_label=0, clean=False):
    model.eval()
    color = args.pretrained_path.lower().find('vgg16')>=0
    _sum = 0
    total_correct_1 = 0
    total_correct_5 = 0
    params = read_noise_config()
    for i, (img, target) in enumerate(valloader):
        # print('PRINT in utils '+str(img.shape))
        # print(type(img))
        if clean == False:
            for i in range(img.shape[0]):
                if color:
                    poison_tag_colored(img[i],params)
                else:
                    poison_tag_bw(img[i],params)
                target[i] = target_label
        if params['model']!='mnist':
            img = torch.squeeze(img) #mnist squeeze 会导致单通道的维度也被去掉，变成28*28

        img, target = img.to(args.device), target.to(args.device)
        output = model(img)
        
        prediction = torch.argsort(output, dim=-1, descending=True)

        total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

        _sum += img.size(0)

    if clean:
        print('epoch:{} top-1 clean accuracy: {:.4f}'.format(epoch, total_correct_1*1.0 / _sum))
        print('epoch:{} top-5 clean accuracy: {:.4f}'.format(epoch, total_correct_5*1.0 / _sum))

    else:
        print('epoch:{} attack accuracy: {:.4f}'.format(epoch, total_correct_1*1.0 / _sum))

    return total_correct_1*1.0 / _sum, total_correct_5*1.0 / _sum


def validate_classes(model, epoch, valloader=None, args=None, criterion=torch.nn.CrossEntropyLoss(), target_label=0, clean=False):
    
    losses = AverageMeter()
    model.eval()
    class_loader = {}
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_test = datasets.CIFAR10('./data', train=False, download=True, transform=trans_cifar)
    for c in range(10):
        class_loader[c] = DataLoader(dataset_test, batch_size=args.bs, sampler=SubsetRandomSampler(np.where(np.array(dataset_test.targets) == c)[0]))

    
    result = display_test(model,class_loader,args,clean=clean)
    res_df = pd.DataFrame.from_dict(result)
    print(res_df)

    acc_list = []
    for k,v in result.items():
        s = sum(v) 
        acc = v[int(k)]/s
        acc_list.append(acc)
    print(acc_list)
    acc = np.mean(acc_list)
    
    if clean==False:
        acc = res_df[2][target_label]/1000

    if clean:
        print('epoch:{} clean accuracy: {:.4f}'.format(epoch, acc))
    else:
        print('epoch:{} attack accuracy: {:.4f}'.format(epoch, acc))

    return acc





def draw_plot(clean_acc, attack_rate, args):
    x = [i for i in range(len(clean_acc))]
    plt.figure(figsize=(9,6))
    plt.plot(x,clean_acc,label='clean acc',marker='^',linewidth=1,markersize=3)
    plt.plot(x,attack_rate,label='attack rate',marker='o',linewidth=1,markersize=3)
    plt.xlabel('Local Epoch')
    plt.ylabel('rate')
    plt.title('fed backdoor')
    plt.legend()
    plt.savefig('./save/fed_backdoor_result/{}_{}_c{}_e{}_Lep{}_alpha{}.png'.format(
        args.dataset,args.model,args.num_users,args.epochs,args.local_ep,args.bd_alpha
    ),dpi=200)


def validate_vit(model, epoch, valloader, criterion, clean):
    losses = AverageMeter()
    model.eval()
    # correct = 0
    _sum = 0
    total_correct_1 = 0
    total_correct_5 = 0

    for i, (img, target, poisoned_flags) in enumerate(valloader):
        img = torch.squeeze(img)
        # target = torch.squeeze(target)
        img, target = img.to(device), target.to(device)
        output = model(img)[0]
        # print(output.shape)
        # print(output)
        prediction = torch.argsort(output, dim=-1, descending=True)

        total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

        # output_np = output.cpu().detach().numpy()
        # target_np = target.cpu().detach().numpy()
        
        # out_ys = np.argmax(output_np, axis = -1)

        # print('out_ys', out_ys)
        # print('target_np', target_np)
        # print('==', out_ys == target_np)
        # sys.exit()

        # _ = out_ys == target_np
        # correct += np.sum(_, axis = -1)
        # _sum += _.shape[0]
        _sum += img.size(0)
        # loss = criterion(output, target)
        # losses.update(loss.item(), img.size(0))

    if clean:
        print('epoch:', epoch)
        # print('clean accuracy: {:.4f}'.format(correct*1.0 / _sum))
        print('top-1 clean accuracy: {:.4f}'.format(total_correct_1*1.0 / _sum))
        print('top-5 clean accuracy: {:.4f}'.format(total_correct_5*1.0 / _sum))
        # print('loss:', losses.avg)
    else:
        print('epoch:', epoch)
        # print('attack accuracy: {:.4f}'.format(correct*1.0 / _sum))
        print('attack accuracy: {:.4f}'.format(total_correct_1*1.0 / _sum))
        # print('loss:', losses.avg)

    # return correct*1.0 / _sum
    return total_correct_1*1.0 / _sum, total_correct_5*1.0 / _sum