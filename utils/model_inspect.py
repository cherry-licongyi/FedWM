import torch
import torchvision

import os
import numpy as np
import matplotlib.pyplot as plt



def load_statedict(param):
    model_dir = param['model_dir']
    client_num = param['client_num']
    ep = param['epoch']

    fc_weight_list,fc_bias_list = [],[]
    model_path = os.path.join(model_dir,'epoch{}_client_{}.pth'.format(ep-1,1))
    ben = torch.load(model_path,map_location=torch.device('cpu'))
    fc_weight = ben['classifier.6.weight'].detach().numpy()
    fc_bias = ben['classifier.6.bias'].detach().numpy()
    fc_weight_list.append(fc_weight)
    fc_bias_list.append(fc_bias)
    
    # fc_weight_list,fc_bias_list = [],[]
    for i in range(client_num):
        model_path = os.path.join(model_dir,'epoch{}_client_{}.pth'.format(ep,i))
        print(model_path)
        msd = torch.load(model_path,map_location=torch.device('cpu'))
        fc_weight = msd['classifier.6.weight'].detach().numpy()
        fc_bias = msd['classifier.6.bias'].detach().numpy()
        fc_weight_list.append(fc_weight)
        fc_bias_list.append(fc_bias)
        # for k,v in m.items():
        #     print(k,v.shape)
    fc_weight_list = np.array(fc_weight_list)
    fc_bias_list = np.array(fc_bias_list)

    return fc_weight_list, fc_bias_list



def visualize(wlist,title):
    # 指定分组个数
    num_bins = 10
    fig, ax = plt.subplots(2, 5, figsize=(20, 8))

    plt.ion()  
    # for i in range(num):
    for j in range(5):
        ax[0][j].clear()
        ax[0][j].hist(wlist[j], num_bins, density=1)
        ax[0][j].set_title('category '+str(j))
        # ax[0][j].set_xlim(0, 0.05)
        ax[1][j].clear()
        ax[1][j].hist(wlist[j+5], num_bins, density=1)
        ax[1][j].set_title('category '+str(j+5))
        # ax[1][j].set_xlim(0, 0.05)


        
        plt.draw()
    plt.pause(1)
    plt.ioff()
    plt.savefig(title) 




def main(param):
    weight_list, bias_list = load_statedict(param)
    print(weight_list.shape, bias_list.shape)

    # Difference with malicious client
    # weight
    # diff_weight_evil = np.abs(weight_list - weight_list[0])[1:]
    # diff_weight_evil = (weight_list - weight_list[0])[1:]
    diff_weight_evil = (weight_list - weight_list[0])[1]
    print(diff_weight_evil.shape)
    # diff_weight_evil = np.mean(diff_weight_evil, axis = 0)
    # print(diff_weight_evil.shape)

    visualize(diff_weight_evil,title='./inspect_result/{}_fcw.png'
              .format(param['model_dir'].split('/')[-1]))
    
    # bias
    # diff_bias_evil = np.abs(bias_list - bias_list[0])[1:]
    # diff_bias_evil = (bias_list - bias_list[0])[1:]
    # print(diff_bias_evil.shape)
    # diff_bias_evil = np.mean(diff_bias_evil, axis = 0)
    # print(diff_bias_evil.shape)

    # visualize(diff_bias_evil,title='./inspect_result/{}_fcb.png'
    #           .format(param['model_dir'].split('/')[-1]))

    # Difference with normal client
    diff_weight_evil = (weight_list - weight_list[0])[2:]
    print(diff_weight_evil.shape)
    diff_weight_evil = np.mean(diff_weight_evil, axis = 0)
    print(diff_weight_evil.shape)

    visualize(diff_weight_evil,title='./inspect_result/{}_n_fcw.png'
              .format(param['model_dir'].split('/')[-1]))





if __name__ == '__main__':
    param = {
        'epoch' : 3,
        'client_num':5,
        'client_evil':0,
        # 'client_trust':
        'model_dir' : '/home/data/gy/Datafree_Backdoor_Fed/federated-learning/checkpoints/backdoor_fed/cifar_vgg16_5'
        
        }
    
    main(param)