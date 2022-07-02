import math

import torch
import numpy as np
from torch.nn import Module, Conv2d, LayerNorm, ReLU, ModuleList, Conv1d, ELU, ZeroPad2d

def get_activation(name):
    if name == 'ReLU':
        return ReLU()
    elif name == 'ELU':
        return ELU()
    raise KeyError(name)

class SeparableConv2d(Module):
    def __init__(self, i, o, k, s, in_F, in_T, fuller=False, activation='ReLU', relu_after_bn=True):
        super(SeparableConv2d, self).__init__()
        # this is actually "same" padding, but PyTorch doesn't support that
        padding = (in_T-1)//s[0] * s[0] + k - in_T
        self.pad1 = ZeroPad2d((padding//2, padding - padding//2, 0, 0))
        self.conv1 = Conv2d(i, o, kernel_size=(1, k), stride=(1, s[0]))
        self.ln1 = LayerNorm((o, in_F, (in_T-1)//s[0]+1))
        self.relu1 = get_activation(activation)
        # this is actually "same" padding, but PyTorch doesn't support that
        padding = (in_F-1)//s[1] * s[1] + k - in_F
        self.pad2 = ZeroPad2d((0, 0, padding//2, padding - padding//2))
        if fuller:
            self.conv2 = Conv2d(o, o, kernel_size=(k, 1), stride=(s[1], 1))
        else:
            self.conv2 = Conv2d(o, o, kernel_size=(k, 1), stride=(s[1], 1), groups=o)
        self.ln2 = LayerNorm((o, (in_F-1)//s[1]+1, (in_T-1)//s[0]+1))
        self.relu2 = get_activation(activation)
        
        self.relu_after_bn = relu_after_bn
        self.hacked = False
    
    # I found a way to do Keras same padding for stride=2 without zero padding layer/function
    # Just flip the image, then the builtin Conv2d padding will do the right thing
    def hack(self):
        self.hacked = not self.hacked
        with torch.no_grad():
            self.conv1.weight.set_(self.conv1.weight.flip([2, 3]))
            self.ln1.weight.set_(self.ln1.weight.flip([1, 2]))
            self.ln1.bias.set_(self.ln1.bias.flip([1, 2]))
            self.conv2.weight.set_(self.conv2.weight.flip([2, 3]))
            self.ln2.weight.set_(self.ln2.weight.flip([1, 2]))
            self.ln2.bias.set_(self.ln2.bias.flip([1, 2]))
            if self.hacked:
                self.conv1.padding = self.pad1.padding[3::-2]
                self.conv2.padding = self.pad2.padding[3::-2]
            else:
                self.conv1.padding = (0,0)
                self.conv2.padding = (0,0)
    
    def forward(self, x):
        if not self.hacked:
            x = self.pad1(x)
        x = self.conv1(x)
        if self.relu_after_bn:
            x = self.ln1(x)
            x = self.relu1(x)
        else:
            x = self.relu1(x)
            x = self.ln1(x)
        if not self.hacked:
            x = self.pad2(x)
        x = self.conv2(x)
        if self.relu_after_bn:
            x = self.ln2(x)
            x = self.relu2(x)
        else:
            x = self.relu2(x)
            x = self.ln2(x)
        return x

class MyF(Module):
    def __init__(self, d, h, u, in_F, in_T, fuller=False, activation='ReLU',
            strides=None, relu_after_bn=True):
        super(MyF, self).__init__()
        channels = [1, d, d, 2*d, 2*d, 4*d, 4*d, h, h]
        convs = []
        for i in range(8):
            k = 3
            s = 2, 2
            if strides is not None:
                s = strides[i][0][1], strides[i][1][0]
            sepconv = SeparableConv2d(channels[i], channels[i+1], k, s, in_F, in_T,
                fuller=fuller,
                activation=activation,
                relu_after_bn=relu_after_bn
            )
            convs.append(sepconv)
            in_F = (in_F-1)//s[1] + 1
            in_T = (in_T-1)//s[0] + 1
        assert in_F==in_T==1, 'output must be 1x1'
        self.convs = ModuleList(convs)
    
    def hack(self):
        for conv in self.convs:
            conv.hack()

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
    
    def forward(self, x, norm=True):
        x = x.reshape([-1, self.h, 1])
        x = self.linear1(x)
        x = self.elu(x)
        x = self.linear2(x)
        x = x.reshape([-1, self.d])
        if norm:
            x = torch.nn.functional.normalize(x, p=2.0)
        return x

class FpNetwork(Module):
    def __init__(self, d, h, u, F, T, params):
        super(FpNetwork, self).__init__()
        self.f = MyF(d, h, u, F, T,
            fuller=params.get('fuller', False),
            activation=params.get('conv_activation', 'ReLU'),
            strides=params.get('strides'),
            relu_after_bn=params.get('relu_after_bn', True)
        )
        self.g = MyG(d, h, u)
        self.hacked = False
    
    def hack(self):
        self.hacked = not self.hacked
        self.f.hack()
    
    def forward(self, x, norm=True):
        if self.hacked:
            x = x.flip([1, 2])
        x = self.f(x)
        x = self.g(x, norm=norm)
        return x
