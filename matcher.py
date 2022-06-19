import csv
import math
import os
import sys
import time
import warnings

import faiss
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
from database import Database

if __name__ == "__main__":
    logger_init = simpleutils.MultiProcessInitLogger('nnmatcher')
    logger_init()
    
    mp.set_start_method('spawn')
    logger = mp.get_logger()
    if len(sys.argv) < 4:
        print('Usage: python %s <query list> <database dir> <result file>' % sys.argv[0])
        sys.exit()
    file_list_for_query = sys.argv[1]
    dir_for_db = sys.argv[2]
    result_file = sys.argv[3]
    result_file2 = os.path.splitext(result_file) # for more detailed output
    result_file2 = result_file2[0] + '_detail.csv'
    result_file_score = result_file + '.bin'
    configs = os.path.join(dir_for_db, 'configs.json')
    params = simpleutils.read_config(configs)
    
    visualize = False

    d = params['model']['d']
    h = params['model']['h']
    u = params['model']['u']
    F_bin = params['n_mels']
    segn = int(params['segment_size'] * params['sample_rate'])
    T = (segn + params['stft_hop'] - 1) // params['stft_hop']
    
    top_k = params['indexer']['top_k']
    frame_shift_mul = params['indexer'].get('frame_shift_mul', 1)

    print('loading model...')
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    model = FpNetwork(d, h, u, F_bin, T, params['model']).to(device)
    model.load_state_dict(torch.load(os.path.join(dir_for_db, 'model.pt'), map_location=device))
    print('model loaded')
    
    print('loading database...')
    db = Database(dir_for_db, params['indexer'], params['hop_size'])
    print('database loaded')

    # doing inference, turn off gradient
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    dataset = MusicDataset(file_list_for_query, params)
    # no task parallelism
    loader = DataLoader(dataset, num_workers=0, batch_size=None)
    
    mel = build_mel_spec_layer(params).to(device)
    
    tm_0 = time.time()
    fout = open(result_file, 'w', encoding='utf8', newline='\n')
    fout2 = open(result_file2, 'w', encoding='utf8', newline='\n')
    fout_score = open(result_file_score, 'wb')
    detail_writer = csv.writer(fout2)
    detail_writer.writerow(['query', 'answer', 'score', 'time', 'part_scores'])
    for dat in tqdm.tqdm(loader):
        embeddings = []
        grads = []
        specs = []
        i, name, wav = dat
        logger.info('get query %s', name)
        tm_1 = time.time()
        i = int(i) # i is leaking file handles!
        
        if wav.shape[0] == 0:
            # load file error!
            logger.error('load %s error!', name)
            ans = 'error'
            sco = -1e999
            tim = 0
            fout.write('%s\t%s\n' % (name, ans))
            fout.flush()
            detail_writer.writerow([name, ans, sco, tim])
            fout2.flush()
            
            song_score = np.zeros([len(db.songList), 2], dtype=np.float32)
            fout_score.write(song_score.tobytes())
            continue
        
        # batch size should be less than 20 because query contains at most 19 segments
        for batch in torch.split(wav, 16):
            g = batch.to(device)
            
            # Mel spectrogram
            with warnings.catch_warnings():
                # torchaudio is still using deprecated function torch.rfft
                warnings.simplefilter("ignore")
                g = mel(g)
            if visualize:
                g.requires_grad = True
            z = model.forward(g, norm=False).cpu()
            if visualize:
                z.backward(z)
                z.detach_()
                grads.append(g.grad.cpu())
                specs.append(g.detach().cpu())
            z = torch.nn.functional.normalize(z, p=2)
            embeddings.append(z)
        embeddings = torch.cat(embeddings)

        tm_2 = time.time()
        logger.info('compute embedding %.6fs', tm_2 - tm_1)

        if visualize:
            grads = torch.cat(grads)
            specs = torch.cat(specs)
        sco, (ans, tim), song_score = db.query_embeddings(embeddings.numpy())
        upsco = []
        ans = db.songList[ans]
        #tim /= frame_shift_mul
        #tim *= params['hop_size']
        #song_score[:, 1] *= params['hop_size'] / frame_shift_mul
        if visualize:
            grads = torch.abs(grads)
            grads = torch.nn.functional.normalize(grads, p=np.inf)
            grads = grads.transpose(0, 1).flatten(1, 2)
            grads = grads.repeat(3, 1, 1)
            specs = specs.transpose(0, 1).flatten(1, 2)
            grads[1] = specs - math.log(1e-6)
            grads[1] /= torch.max(grads[1])
            grads[0] = torch.nn.functional.relu(grads[0])
            grads[1] *= 1 - grads[0]
            grads[2] = 0
            grads = torch.flip(grads, [1])
            grads[:,:,::32] = 0
            torchvision.utils.save_image(grads, '%s.png' % os.path.basename(name[0]))

        tm_1 = time.time()
        fout.write('%s\t%s\n' % (name, ans))
        fout.flush()
        detail_writer.writerow([name, ans, sco, tim] + upsco)
        fout2.flush()
        
        fout_score.write(song_score.tobytes())
        tm_2 = time.time()
        logger.info('output answer %.6fs', tm_2 - tm_1)
    fout.close()
    fout2.close()
    logger.info('total query time %.6fs', time.time() - tm_0)
else:
    torch.set_num_threads(1)
