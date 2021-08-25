import csv
import ctypes
from ctypes import cdll, c_float, c_int, c_int64, c_void_p, POINTER
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

cpp_accelerate = False
gpu_accelerate = False
if cpp_accelerate:
    mydll = cdll.LoadLibrary('cpp/seqscore')
    mydll.seq_score.argtypes = [
        c_void_p,
        POINTER(c_int64),
        c_int,
        POINTER(c_float),
        c_int,
        POINTER(c_int64),
        c_int,
        POINTER(c_float),
        c_int,
        c_int
    ]
    mydll.seq_score.restype = c_int

def query_embeddings(index_gpu, query, k, song_pos, index_cpu, frame_shift_mul):
    '''論文進度 30%'''
    logger = mp.get_logger()
    tm_1 = time.time()
    d = index.d
    distances, labels = index_gpu.search(query, k)
    tm_2 = time.time()
    best = -1e999
    best_song_t = -1, 0
    song_score = np.zeros([song_pos.shape[0] - 1, 2], dtype=np.float32)
    if index_gpu.ntotal == 0:
        return best, best_song_t, song_score
    
    for shift in range(frame_shift_mul):
        candidates = []
        subquery = query[shift::frame_shift_mul]
        sub_len = subquery.shape[0]
        for t in range(sub_len):
            lab = labels[t * frame_shift_mul + shift]
            lab = lab[lab != -1]
            song_id = np.searchsorted(song_pos, lab, side='right') - 1
            song_t = lab - song_pos[song_id] - t
            candidates.append(np.stack([song_id, song_t], axis=1))
        # according to NumPy, np.unique returns sorted array
        candidates = np.unique(np.concatenate(candidates), axis=0)
        
        vec = np.zeros_like(subquery)
        for c in candidates:
            song_id = c[0].item()
            song_start = song_pos[song_id].item()
            song_len = song_pos[song_id+1].item() - song_start
            t = c[1].item()
            
            # get corresponding embeddings from db
            for i in range(sub_len):
                if t+i < 0 or t+i >= song_len:
                    vec[i] = 0.0
                else:
                    index_cpu.reconstruct(song_start + t+i, vec[i])
            # compute average score
            sco = np.dot(vec.flatten(), subquery.flatten()).item() / sub_len
            if sco > song_score[song_id, 0]:
                song_score[song_id, 0] = sco
                song_score[song_id, 1] = t * frame_shift_mul + shift
            if sco > best:
                best = sco
                best_song_t = song_id, t * frame_shift_mul + shift
    tm_3 = time.time()
    logger.info('search %.6fs rerank %.6fs', tm_2-tm_1, tm_3-tm_2)
    return best, best_song_t, song_score

def query_embeddings_cpp(index_gpu, query, k, song_pos, index_cpu, frame_shift_mul):
    logger = mp.get_logger()
    tm_1 = time.time()
    d = index.d
    distances, labels = index_gpu.search(query, k)
    tm_2 = time.time()
    best = -1e999
    best_song_t = -1, 0
    song_score = np.zeros([song_pos.shape[0] - 1, 2], dtype=np.float32)

    for shift in range(frame_shift_mul):
        subquery = np.ascontiguousarray(query[shift::frame_shift_mul])
        sublabel = np.ascontiguousarray(labels[shift::frame_shift_mul])
        song_id = mydll.seq_score(
            int(index_cpu.this),
            song_pos.ctypes.data_as(POINTER(c_int64)),
            song_pos.shape[0]-1,
            subquery.ctypes.data_as(POINTER(c_float)),
            subquery.shape[0],
            sublabel.ctypes.data_as(POINTER(c_int64)),
            k,
            song_score.ctypes.data_as(POINTER(c_float)),
            shift,
            frame_shift_mul
        )
        sco = song_score[song_id, 0].item()
        if sco > best:
            best = sco
            best_song_t = song_id, song_score[song_id, 1].item()
    tm_3 = time.time()
    logger.info('search %.6fs rerank %.6fs', tm_2-tm_1, tm_3-tm_2)
    return best, best_song_t, song_score

if cpp_accelerate:
    query_embeddings = query_embeddings_cpp

if __name__ == "__main__":
    logger_init = simpleutils.MultiProcessInitLogger('matcher')
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
    device = torch.device('cuda')
    model = FpNetwork(d, h, u, F_bin, T, params['model']).to(device)
    model.load_state_dict(torch.load(os.path.join(dir_for_db, 'model.pt')))
    print('model loaded')
    
    print('loading database...')
    songList = simpleutils.read_file_list(os.path.join(dir_for_db, 'songList.txt'))
    
    landmarkKey = np.fromfile(os.path.join(dir_for_db, 'landmarkKey'), dtype=np.int32)
    index = faiss.read_index(os.path.join(dir_for_db, 'landmarkValue'))
    if hasattr(index, 'make_direct_map'):
        index.make_direct_map()
    assert len(songList) == landmarkKey.shape[0]
    index2song = np.repeat(np.arange(len(songList)), landmarkKey)
    landmarkKey = np.pad(np.cumsum(landmarkKey, dtype=np.int64), (1, 0))
    print('database loaded')
    if isinstance(index, faiss.IndexIVF):
        print('inverse list count:', index.nlist)
        index.nprobe = params['indexer'].get('nprobe', 50)
        print('num probes:', index.nprobe)

    if gpu_accelerate:
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = True
        gpu_index = faiss.index_cpu_to_all_gpus(index, co, 1)
    else:
        gpu_index = index

    # doing inference, turn off gradient
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    dataset = MusicDataset(file_list_for_query, params)
    # no task parallelism
    loader = DataLoader(dataset, num_workers=0, batch_size=None)
    
    mel = build_mel_spec_layer(params).to(device)
    
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
            
            song_score = np.zeros([len(songList), 2], dtype=np.float32)
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
        sco, (ans, tim), song_score = query_embeddings(gpu_index, embeddings.numpy(), top_k, landmarkKey, index, frame_shift_mul)
        upsco = []
        ans = songList[ans]
        tim /= frame_shift_mul
        tim *= params['hop_size']
        song_score[:, 1] *= params['hop_size'] / frame_shift_mul
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
else:
    torch.set_num_threads(1)
