import torch
import torch.nn as nn
from typing import List, Tuple, Any
import torch.utils.data as data
import torch.nn.functional as F

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

class MNIST_L5(nn.Module):
    def __init__(self, dropout=0.0):
        super(MNIST_L5, self).__init__()

        self.dropout = dropout

        self.block = nn.Sequential(
            nn.Conv2d(1, 32, 2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 2),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(128 * 5**2 , 200)
        self.fc2 = nn.Linear(200, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout(x)
        out = self.block(x)
        out = out.view(-1,  128 * 5**2)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return F.log_softmax(out,1)

def mnist_l5():
    return MNIST_L5()
# print("success")