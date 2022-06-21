import os
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import tqdm

import simpleutils
from model import FpNetwork
from datautil.melspec import build_mel_spec_layer
from datautil.musicdata import MusicDataset

if __name__ == "__main__":
    logger_init = simpleutils.MultiProcessInitLogger('nnextract')
    logger_init()
    
    mp.set_start_method('spawn')
    if len(sys.argv) < 4:
        print('Usage: python %s <query list> <database dir> <output embedding dir>' % sys.argv[0])
        sys.exit()
    file_list_for_query = sys.argv[1]
    dir_for_db = sys.argv[2]
    out_embed_dir = sys.argv[3]
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
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    model = FpNetwork(d, h, u, F_bin, T, params['model']).to(device)
    model.load_state_dict(torch.load(os.path.join(dir_for_db, 'model.pt'), map_location=device))
    print('model loaded')
    
    # doing inference, turn off gradient
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    dataset = MusicDataset(file_list_for_query, params)
    loader = DataLoader(dataset, num_workers=4, batch_size=None, worker_init_fn=logger_init)
    
    mel = build_mel_spec_layer(params).to(device)
    
    os.makedirs(out_embed_dir, exist_ok=True)
    embeddings_file = open(os.path.join(out_embed_dir, 'query_embeddings'), 'wb')
    query_idx = open(os.path.join(out_embed_dir, 'query_index'), 'wb')
    tm_0 = time.time()
    idx_pos = 0
    for dat in tqdm.tqdm(loader):
        logger = mp.get_logger()
        i, name, wav = dat
        logger.info('get query %s', name)
        tm_1 = time.time()
        i = int(i) # i is leaking file handles!
        
        if wav.shape[0] == 0:
            # load file error!
            logger.error('load %s error!', name)

            query_idx.write(np.array([idx_pos, 0], dtype=np.int64))
            continue
        
        idx_start = idx_pos
        # batch size should be less than 20 because query contains at most 19 segments
        for batch in torch.split(wav, 16):
            g = batch.to(device)
            
            # Mel spectrogram
            g = mel(g)
            z = model(g).cpu()
            embeddings_file.write(z.numpy().tobytes())
            idx_pos += z.shape[0]
        query_idx.write(np.array([idx_start, idx_pos - idx_start], dtype=np.int64))

        tm_2 = time.time()
        logger.info('compute embedding %.6fs', tm_2 - tm_1)
    embeddings_file.flush()
    print('total', idx_pos, 'embeddings')
    shutil.copyfile(file_list_for_query, os.path.join(out_embed_dir, 'queryList.txt'))

    # write settings
    shutil.copyfile(configs, os.path.join(out_embed_dir, 'configs.json'))
    
    logger.info('total extract time %.6fs', time.time() - tm_0)
