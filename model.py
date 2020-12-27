import math

import torch
import numpy as np
from tqdm import trange
from torch.nn import Module, Conv2d, LayerNorm, ReLU, ModuleList, Conv1d, ELU, ZeroPad2d

class SeparableConv2d(Module):
    def __init__(self, i, o, k, s, in_F, in_T):
        super(SeparableConv2d, self).__init__()
        # this is actually "same" padding, but PyTorch doesn't support that
        padding = (in_T-1)//s * s + k - in_T
        self.pad1 = ZeroPad2d((padding//2, padding - padding//2, 0, 0))
        self.conv1 = Conv2d(i, o, kernel_size=(1, k), stride=(1, s))
        self.ln1 = LayerNorm((o, in_F, (in_T-1)//s+1))
        self.relu1 = ReLU()
        # this is actually "same" padding, but PyTorch doesn't support that
        padding = (in_F-1)//s * s + k - in_F
        self.pad2 = ZeroPad2d((0, 0, padding//2, padding - padding//2))
        self.conv2 = Conv2d(o, o, kernel_size=(k, 1), stride=(s, 1), groups=o)
        self.ln2 = LayerNorm((o, (in_F-1)//s+1, (in_T-1)//s+1))
        self.relu2 = ReLU()
    
    def forward(self, x):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.relu1(x)
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.ln2(x)
        x = self.relu2(x)
        return x

class MyF(Module):
    def __init__(self, d, h, u, in_F, in_T):
        super(MyF, self).__init__()
        channels = [1, d, d, 2*d, 2*d, 4*d, 4*d, h, h]
        convs = []
        for i in range(8):
            k = 3
            s = 2
            convs.append(SeparableConv2d(channels[i], channels[i+1], k, s, in_F, in_T))
            in_F = (in_F-1)//s + 1
            in_T = (in_T-1)//s + 1
        assert in_F==in_T==1, 'output must be 1x1'
        self.convs = ModuleList(convs)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        for i, conv in enumerate(self.convs):
            x = conv(x)
        #assert x.shape[2]==x.shape[3]==1, 'output must be 1x1'
        return x

class MyG(Module):
    __constants__ = ['d', 'h']
    def __init__(self, d, h, u):
        super(MyG, self).__init__()
        assert h%d == 0, 'h must be divisible by d'
        v = h//d
        self.d = d
        self.h = h
        self.u = u
        self.v = v
        self.linear1 = Conv1d(d * v, d * u, kernel_size=(1,), groups=d)
        self.elu = ELU()
        self.linear2 = Conv1d(d * u, d, kernel_size=(1,), groups=d)
    
    def forward(self, x):
        x = x.reshape([-1, self.h, 1])
        x = self.linear1(x)
        x = self.elu(x)
        x = self.linear2(x)
        x = x.reshape([-1, self.d])
        x = torch.nn.functional.normalize(x, p=2.0)
        return x

class FpNetwork(Module):
    def __init__(self, d, h, u, F, T):
        super(FpNetwork, self).__init__()
        self.f = MyF(d, h, u, F, T)
        self.g = MyG(d, h, u)
    
    def forward(self, x):
        x = self.f(x)
        x = self.g(x)
        return x
