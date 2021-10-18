import argparse
import datetime
import os
import shutil
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import tensorboardX
import torch_optimizer as optim

from model import FpNetwork
from datautil.dataset_v2 import SegmentedDataLoader
from datautil.mock_data import make_false_data
import simpleutils
from datautil.specaug import SpecAugment

from torch.cuda.amp import autocast, GradScaler

# fix PyTorch bug #49630
# apply pull request #49631
CosineAnnealingWarmRestarts = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
def new_cosinedecay_init(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min

        self.T_cur = 0 if last_epoch < 0 else last_epoch
        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch, verbose)

torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.__init__ = new_cosinedecay_init

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

def train(model, optimizer, train_data, val_data, batch_size, device, params, writer, start_epoch, scaler):
    logger = mp.get_logger()
    minibatch = 40
    if torch.cuda.get_device_properties(0).total_memory > 11e9:
        minibatch = 640
    total_epoch = params.get('epoch', 100)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
            T_0=total_epoch, eta_min=1e-7, last_epoch=start_epoch)
    os.makedirs(params['model_dir'], exist_ok=True)
    specaug = SpecAugment(params)
    for epoch in range(start_epoch+1, total_epoch):
        logger.info('epoch %d', epoch+1)
        model.train()
        tau = params.get('tau', 0.05)
        print('epoch %d' % (epoch+1))
        losses = []
        # set dataloadet to train mode
        train_data.shuffle = True
        train_data.eval_time_shift = False
        train_data.augmented = True
        train_data.set_epoch(epoch)

        pbar = tqdm(train_data, ncols=80)
        for x in pbar:
            optimizer.zero_grad()
        
            x = torch.flatten(x, 0, 1)
            x = specaug.augment(x)
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
                with autocast():
                    y = model(x.to(device))
                    loss = similarity_loss(y, tau)
                scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lossnum = float(loss.item())
            pbar.set_description('loss=%f'%lossnum)
            losses.append(lossnum)
        writer.add_scalar('train/loss', np.mean(losses), epoch)
        print('loss: %f' % np.mean(losses))

        model.eval()
        with torch.no_grad():
            print('validating')
            x_embed = []
            # set dataloader to eval mode
            train_data.shuffle = False
            train_data.eval_time_shift = True
            train_data.augmented = False

            for x in tqdm(train_data, desc='train data', ncols=80):
                x = x[:, 0]
                for xx in torch.split(x, minibatch):
                    y = model(xx.to(device)).cpu()
                    x_embed.append(y)
            x_embed = torch.cat(x_embed)
            train_N = x_embed.shape[0]
            acc = 0
            validate_N = 0
            y_embed = []
            for x in tqdm(val_data, desc='val data', ncols=80):
                x = torch.flatten(x, 0, 1)
                for xx in torch.split(x, minibatch):
                    y = model(xx.to(device)).cpu()
                    y_embed.append(y)
            y_embed = torch.cat(y_embed)
            y_embed_org = y_embed[0::2]
            y_embed_aug = y_embed[1::2].to(device)
            
            # compute validation score on GPU
            self_score = []
            for embeds in torch.split(y_embed_org, 320):
                A = torch.matmul(y_embed_aug, embeds.T.to(device))
                self_score.append(A.diagonal(-validate_N).cpu())
                validate_N += embeds.shape[0]
            self_score = torch.cat(self_score).to(device)
            
            ranks = torch.zeros(validate_N, dtype=torch.long).to(device)
            for embeds in torch.split(x_embed, 320):
                A = torch.matmul(y_embed_aug, embeds.T.to(device))
                ranks += (A.T >= self_score).sum(dim=0)
            for embeds in torch.split(y_embed_org, 320):
                A = torch.matmul(y_embed_aug, embeds.T.to(device))
                ranks += (A.T >= self_score).sum(dim=0)
            acc = int((ranks == 1).sum())
            acc10 = int((ranks <= 10).sum())
            acc20 = int((ranks <= 20).sum())
            acc100 = int((ranks <= 100).sum())
            print('validate score: %f' % (acc / validate_N,))
            writer.add_scalar('validation/accuracy', acc / validate_N, epoch)
            writer.add_scalar('validation/top10', acc10 / validate_N, epoch)
            writer.add_scalar('validation/top20', acc20 / validate_N, epoch)
            writer.add_scalar('validation/top100', acc100 / validate_N, epoch)
            #writer.add_scalar('validation/MRR', (1/ranks).mean(), epoch)
        scheduler.step()
        del A, ranks, self_score, y_embed_aug, y_embed_org, y_embed
        writer.flush()
        
        # save checkpoint
        check = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
        }
        torch.save(check, os.path.join(params['model_dir'], 'checkpoint%d.ckpt' % epoch))
        # cleanup old checkpoints
        if epoch % 10 != 0:
            try:
                os.unlink(os.path.join(params['model_dir'], 'checkpoint%d.ckpt' % (epoch-10)))
            except:
                pass
        with open(os.path.join(params['model_dir'], 'epochs.txt'), 'w') as fout:
            fout.write('%d\n' % epoch)
    os.makedirs(params['model_dir'], exist_ok=True)
    torch.save(model.state_dict(), os.path.join(params['model_dir'], 'model.pt'))

def test_train(args):
    logger = mp.get_logger()
    params = simpleutils.read_config(args.params)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    d = params['model']['d']
    h = params['model']['h']
    u = params['model']['u']
    F_bin = params['n_mels']
    segn = int(params['segment_size'] * params['sample_rate'])
    T = (segn + params['stft_hop'] - 1) // params['stft_hop']
    batch_size = params['batch_size']
    device = torch.device('cuda')
    model = FpNetwork(d, h, u, F_bin, T, params['model']).to(device)
    
    optimizer = params.get('optimizer', 'adam')
    if optimizer == 'lamb':
        optimizer = optim.Lamb(model.parameters(), lr=params.get('lr', 1e-4),
            weight_decay=1e-6, clamp_value=1e3, debias=True)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=params.get('lr', 1e-4))
    scaler = GradScaler()
    
    # load checkpoint
    os.makedirs(params['model_dir'], exist_ok=True)
    epoch = -1
    if os.path.exists(os.path.join(params['model_dir'], 'date.txt')):
        with open(os.path.join(params['model_dir'], 'date.txt')) as fin:
            date_str = next(fin).strip()
    else:
        date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        with open(os.path.join(params['model_dir'], 'date.txt'), 'w') as fout:
            fout.write(date_str + '\n')
    
    if os.path.exists(os.path.join(params['model_dir'], 'epochs.txt')):
        with open(os.path.join(params['model_dir'], 'epochs.txt')) as fin:
            epoch = int(fin.read().strip())
        if epoch+1 >= params.get('epoch', 100):
            print('This model has finished training!')
            exit(1)
        print('Load from epoch %d' % (epoch+1))
        check = torch.load(os.path.join(params['model_dir'], 'checkpoint%d.ckpt' % epoch))
        model.load_state_dict(check['model'])
        optimizer.load_state_dict(check['optimizer'])
        if 'scaler' in check:
            scaler.load_state_dict(check['scaler'])
    else:
        shutil.copyfile(args.params, os.path.join(params['model_dir'], 'configs.json'))
    
    # tensorboard visualize
    safe_name = os.path.split(params['model_dir'])[1]
    if safe_name == '':
        safe_name = os.path.split(os.path.split(params['model_dir'])[0])[1]
    log_dir = "runs/" + safe_name + '-' + date_str
    writer = tensorboardX.SummaryWriter(log_dir)
    
    if torch.cuda.is_available():
        print('GPU mem usage: %dMB' % (torch.cuda.memory_allocated()/1024**2))

    logger.info('load augmentation data')
    train_data = SegmentedDataLoader('train', params, num_workers=args.workers)
    print('training data contains %d samples' % len(train_data.dataset))
    
    val_data = SegmentedDataLoader('validate', params, num_workers=args.workers)
    val_data.shuffle = False
    val_data.eval_time_shift = True
    print('validation data contains %d samples' % len(val_data.dataset))
    
    train(model, optimizer, train_data, val_data, batch_size, device, params, writer, epoch, scaler)

if __name__ == "__main__":
    logger_init = simpleutils.MultiProcessInitLogger('train')
    logger_init()
    logger = mp.get_logger()
    logger.info('logger init')
    torch.use_deterministic_algorithms(True)
    mp.set_start_method('spawn')
    args = argparse.ArgumentParser()
    args.add_argument('-p', '--params', default='configs/default.json')
    args.add_argument('-w', '--workers', type=int, default=4)
    args = args.parse_args()
    logger.info(args)
    test_train(args)
