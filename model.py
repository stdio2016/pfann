import torch
import math
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
        assert x.shape[2]==x.shape[3]==1, 'output must be 1x1'
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

def make_false_data(N):
    mock = torch.rand([N, F_bin, T], dtype=torch.float32)
    mock2 = mock + torch.rand([N, F_bin, T], dtype=torch.float32) * 1 - 0.5
    mock = torch.stack([mock, mock2], dim=1)
    mock = mock.reshape([-1, F_bin, T])
    return mock

def similarity_loss(y, tau):
    a = torch.matmul(y, y.T)
    a /= tau
    Ls = []
    for i in range(y.shape[0]):
        nn_self = torch.cat([a[i,:i], a[i,i+1:]])
        softmax = torch.nn.functional.log_softmax(nn_self, dim=0)
        Ls.append(softmax[i if i%2 == 0 else i-1])
    Ls = torch.stack(Ls)
    
    loss = torch.sum(Ls) / -y.shape[0]
    return loss

d = 128
h = 1024
u = 32
F_bin = 256
T = 32
N = 320
data_N = 2000
validate_N = 100
device = torch.device('cuda')
model = FpNetwork(d, h, u, F_bin, T).to(device)
if torch.cuda.is_available():
    print('GPU mem usage: %dMB' % (torch.cuda.memory_allocated()/1024**2))
x_mock = make_false_data(data_N)
y_mock = make_false_data(validate_N)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    model.train()
    tau = 0.05
    print('epoch %d' % (epoch+1))
    losses = []
    for batch_idx in trange((data_N*2-1)//N+1):
        optimizer.zero_grad()
        
        x = x_mock[N*batch_idx:N*(batch_idx+1)].to(device)
        y = model(x)
        loss = similarity_loss(y, tau)
        loss.backward()
        
        optimizer.step()
        losses.append(float(loss.item()))
    print('loss: %f' % np.mean(losses))

    model.eval()
    with torch.no_grad():
        x_embed =[]
        for batch_idx in trange((data_N*2-1)//N+1):
            x = x_mock[N*batch_idx:N*(batch_idx+1)].to(device)
            y = model(x).cpu()
            x_embed.append(y)
        x_embed = torch.cat(x_embed)
        y_embed = model(y_mock.to(device)).cpu()
        A = torch.matmul(y_embed, torch.cat([x_embed, y_embed]).T)
        ans = torch.topk(A, 2, dim=1)
        acc = 0
        for i in range(y_embed.shape[0]):
            part = i+1 if i%2==0 else i-1
            part += data_N*2
            if ans.indices[i,1] == part:
                acc += 1
        print('validate score: %f' % (acc / (validate_N*2)))

#import subprocess
#subprocess.run(['nvidia-smi'])
