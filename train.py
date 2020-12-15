import numpy as np
from tqdm import tqdm
import torch
from model import FpNetwork
from datautil.mock_data import make_false_data
from torch.utils.data import DataLoader

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

def train(model, optimizer, train_data, val_data, batch_size, device):
    minibatch = 40
    train_data = DataLoader(train_data, batch_size=batch_size//2, shuffle=True)
    validate_N = val_data.shape[0] * 2
    val_data = DataLoader(val_data, batch_size=minibatch//2)
    for epoch in range(100):
        model.train()
        tau = 0.05
        print('epoch %d' % (epoch+1))
        losses = []
        for x in tqdm(train_data):
            optimizer.zero_grad()
        
            x = torch.flatten(x, 0, 1)
            if minibatch < batch_size:
                with torch.no_grad():
                    xs = torch.split(x, minibatch)
                    ys = []
                    for xx in xs:
                        ys.append(model(xx.to(device)))
                # compute gradient of model output
                y = torch.cat(ys)
                y.requires_grad = True
                loss = similarity_loss(y, tau)
                loss.backward()
                # manual backward
                ys = torch.split(y.grad, minibatch)
                for xx, yg in zip(xs, ys):
                    yy = model(xx.to(device))
                    yy.backward(yg.to(device))
            else:
                y = model(x.to(device))
                loss = similarity_loss(y, tau)
                loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        print('loss: %f' % np.mean(losses))

        model.eval()
        with torch.no_grad():
            x_embed = []
            for x in tqdm(train_data):
                x = torch.flatten(x, 0, 1)
                for xx in torch.split(x, minibatch):
                    y = model(xx.to(device)).cpu()
                    x_embed.append(y)
            x_embed = torch.cat(x_embed)
            acc = 0
            for x in tqdm(val_data):
                x = torch.flatten(x, 0, 1).to(device)
                y_embed = model(x).cpu()
                A = torch.matmul(y_embed, torch.cat([x_embed, y_embed]).T)
                ans = torch.topk(A, 2, dim=1)
                for i in range(y_embed.shape[0]):
                    part = i+1 if i%2==0 else i-1
                    part += x_embed.shape[0]
                    if ans.indices[i,1] == part:
                        acc += 1
            print('validate score: %f' % (acc / validate_N))

def test_train():
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    d = 128
    h = 1024
    u = 32
    F_bin = 256
    T = 32
    N = 320
    data_N = 2000
    validate_N = 160
    device = torch.device('cuda')
    model = FpNetwork(d, h, u, F_bin, T).to(device)
    if torch.cuda.is_available():
        print('GPU mem usage: %dMB' % (torch.cuda.memory_allocated()/1024**2))
    x_mock = make_false_data(data_N, F_bin, T)
    y_mock = make_false_data(validate_N, F_bin, T)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(model, optimizer, x_mock, y_mock, N, device)

if __name__ == "__main__":
    test_train()
