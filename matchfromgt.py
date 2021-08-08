import csv
import math
import os
import sys
import argparse
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchvision
import tqdm

# torchaudio currently (0.7) will throw warning that cannot be disabled
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import torchaudio

import simpleutils
from model import FpNetwork
from datautil.melspec import build_mel_spec_layer
from datautil.musicdata import MusicDataset

if __name__ == "__main__":
    mp.set_start_method('spawn')
    args = argparse.ArgumentParser()
    args.add_argument('file_list')
    args.add_argument('gt')
    args.add_argument('db')
    args.add_argument('result')
    args = args.parse_args()
    
    file_list_for_query = args.file_list
    dir_for_db = args.db
    result_file = args.result
    configs = os.path.join(dir_for_db, 'configs.json')
    params = simpleutils.read_config(configs)

    d = params['model']['d']
    h = params['model']['h']
    u = params['model']['u']
    F_bin = params['n_mels']
    segn = int(params['segment_size'] * params['sample_rate'])
    T = (segn + params['stft_hop'] - 1) // params['stft_hop']
    
    frame_shift_mul = params['indexer'].get('frame_shift_mul', 1)

    print('loading model...')
    device = torch.device('cuda')
    model = FpNetwork(d, h, u, F_bin, T, params['model']).to(device)
    model.load_state_dict(torch.load(os.path.join(dir_for_db, 'model.pt')))
    print('model loaded')
    
    print('loading database...')
    with open(os.path.join(dir_for_db, 'songList.txt'), 'r', encoding='utf8') as fin:
        songList = []
        for line in fin:
            if line.endswith('\n'): line = line[:-1]
            songList.append(line)
    
    landmarkKey = np.fromfile(os.path.join(dir_for_db, 'landmarkKey'), dtype=np.int32)
    assert len(songList) == landmarkKey.shape[0]
    index2song = np.repeat(np.arange(len(songList)), landmarkKey)
    landmarkKey = np.pad(np.cumsum(landmarkKey, dtype=np.int64), (1,0))
    songEmb = np.fromfile(os.path.join(dir_for_db, 'embeddings'), dtype=np.float32)
    songEmb = songEmb.reshape([-1, d])
    songEmb = torch.from_numpy(songEmb)
    print('database loaded')

    print('loading ground truth...')
    songList_noext = [os.path.splitext(os.path.basename(x))[0] for x in songList]
    with open(args.gt, 'r', encoding='utf8') as fin:
        gt = {}
        for i in fin:
            query, ans = i.split('\t')
            ans = ans.rstrip()
            gt[query] = songList_noext.index(ans)
    print('ground truth loaded')

    # doing inference, turn off gradient
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    dataset = MusicDataset(file_list_for_query, params)
    # no task parallelism
    loader = DataLoader(dataset, num_workers=0)
    
    mel = build_mel_spec_layer(params).to(device)
    
    fout = open(result_file, 'w', encoding='utf8', newline='\n')
    detail_writer = csv.writer(fout)
    detail_writer.writerow(['query', 'answer', 'score', 'time', 'part_scores'])
    for dat in tqdm.tqdm(loader):
        embeddings = []
        grads = []
        specs = []
        i, name, wav = dat
        i = int(i) # i is leaking file handles!
        
        # get song name
        query = os.path.splitext(os.path.basename(name[0]))[0]
        
        
        if query not in gt:
            print('query %s does not have ground truth' % query)
            continue
        ansId = gt[query]
        ans = songList[ansId]
        
        # batch size should be less than 20 because query contains at most 19 segments
        for batch in DataLoader(wav.squeeze(0), batch_size=16):
            g = batch.to(device)
            
            # Mel spectrogram
            with warnings.catch_warnings():
                # torchaudio is still using deprecated function torch.rfft
                warnings.simplefilter("ignore")
                g = mel(g)
            z = model.forward(g, norm=False).cpu()
            z = torch.nn.functional.normalize(z, p=2)
            embeddings.append(z)
        embeddings = torch.cat(embeddings)
        
        idx1 = landmarkKey[ansId]
        idx2 = landmarkKey[ansId+1]
        T = (embeddings.shape[0]-1) // frame_shift_mul + 1
        slen = idx2 - idx1
        # find alignment
        scos = embeddings @ songEmb[idx1:idx2].T
        accum_scos = torch.zeros([frame_shift_mul, slen + T])
        for t in range(embeddings.shape[0]):
            t0 = T - t//frame_shift_mul
            accum_scos[t % frame_shift_mul, t0:t0+slen] += scos[t]
        # these are invalid time shifts
        accum_scos[:, 0] = -T*2
        accum_scos[(embeddings.shape[0]-1)%frame_shift_mul+1:, 1] = -T*2
        
        tim = torch.argmax(accum_scos).item()
        tim1, tim2 = divmod(tim, slen + T)
        tim = -tim1 + (tim2-T) * frame_shift_mul
        
        tim /= frame_shift_mul
        tim *= params['hop_size']
        sco = accum_scos[tim1, tim2].item()
        myscos = []
        myvecs = []
        tidxs = []
        for t in range(T):
            tidx = t*frame_shift_mul + tim1
            if 0 <= tidx < embeddings.shape[0] and 0 <= tim2-T+t < slen:
                mysco = scos[tidx, tim2-T + t].item()
                tidxs.append(tidx)
                myscos.append(mysco)
                myvecs.append(embeddings[tidx])
        myvecs = torch.stack(myvecs)
        score_seg = myvecs @ songEmb.T

        upsco = []
        for i in range(len(myscos)):
            score_seg[i, idx1 + (tim2-T) + i] = -10
            rank = (score_seg[i] >= myscos[i]).sum().item() + 1
            upsco += [myscos[i], tidxs[i], rank]
        
        detail_writer.writerow([name[0], ans, sco, tim] + upsco)
        fout.flush()
        del score_seg
    fout.close()
else:
    torch.set_num_threads(1)
