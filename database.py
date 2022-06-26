import ctypes
from ctypes import cdll, c_float, c_int, c_int64, c_void_p, POINTER
import os
import time

import faiss
import numpy as np
import torch.multiprocessing as mp

import simpleutils

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
        c_float
    ]
    mydll.seq_score.restype = c_int
    mydll.version.restype = c_int64
    if mydll.version() != 20220625002:
        print('seqscore.cpp Wrong version! Please recompile')
        exit(1)


def make_direct_map(index):
    if isinstance(index, faiss.Index):
        index = faiss.downcast_index(index)
    elif isinstance(index, faiss.IndexBinary):
        index = faiss.downcast_IndexBinary(index)
    if hasattr(index, 'make_direct_map'):
        index.make_direct_map()
        return True
    elif isinstance(index, faiss.IndexPreTransform):
        return make_direct_map(index.index)
    elif isinstance(index, faiss.IndexFlat):
        return True
    else:
        print(type(index), 'does not support direct map yet!')
        return False

def set_search_params(index, params):
    def helper(subindex, subparam):
        for name in subparam:
            value = subparam[name]
            if hasattr(subindex, name):
                if isinstance(value, dict):
                    helper(getattr(subindex, name), value)
                else:
                    setattr(subindex, name, value)
            else:
                print(subindex, 'has no attribute', name)
    if 'search_params' in params:
        helper(index, params['search_params'])

    # set nprobes
    myindex = index
    if isinstance(myindex, faiss.IndexPreTransform):
        myindex = faiss.downcast_index(myindex.index)
    if isinstance(myindex, faiss.IndexIVF):
        print('inverse list count:', myindex.nlist)
        myindex.nprobe = params.get('nprobe', 50)
        print('num probes:', myindex.nprobe)

class Database:
    def __init__(self, dir_for_db, indexer_params, hop_size):
        self.dir_for_db = dir_for_db
        self.params = indexer_params
        self.top_k = self.params['top_k']
        self.frame_shift_mul = self.params.get('frame_shift_mul', 1)
        self.hop_size = hop_size

        self.songList = simpleutils.read_file_list(os.path.join(dir_for_db, 'songList.txt'))
        
        self.song_pos = np.fromfile(os.path.join(dir_for_db, 'landmarkKey'), dtype=np.int32)
        assert len(self.songList) == self.song_pos.shape[0]
        self.song_pos = np.pad(np.cumsum(self.song_pos, dtype=np.int64), (1, 0))

        self.index = faiss.read_index(os.path.join(dir_for_db, 'landmarkValue'))
        try:
            self.embedding = None
            if self.index.ntotal > 0:
                self.index.reconstruct(0)
        except RuntimeError:
            if not make_direct_map(self.index):
                print('This index cannot recover vector')
                self.embedding = np.fromfile(os.path.join(dir_for_db, 'embeddings'), dtype=np.float32)
                self.embedding = self.embedding.reshape([-1, self.index.d])

        set_search_params(self.index, self.params)
        
        if gpu_accelerate and self.params.get('gpu', False):
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            self.gpu_index = faiss.index_cpu_to_all_gpus(self.index, co, 1)
        else:
            self.gpu_index = self.index
        logger = mp.get_logger()
        self.score_alpha = self.params.get('score_alpha', 0)
        logger.info('score alpha: %d', self.score_alpha)
    
    def query_embeddings(self, query):
        if cpp_accelerate:
            return self.query_embeddings_cpp(query)
        else:
            return self.query_embeddings_base(query)

    def query_embeddings_base(self, query):
        logger = mp.get_logger()
        tm_1 = time.time()
        d = self.index.d
        distances, labels = self.gpu_index.search(query, self.top_k)
        tm_2 = time.time()
        best = -1e999
        best_song_t = -1, 0
        song_score = np.zeros([len(self.songList), 2], dtype=np.float32)
        if self.gpu_index.ntotal == 0:
            return best, best_song_t, song_score
        
        for shift in range(self.frame_shift_mul):
            candidates = []
            subquery = query[shift::self.frame_shift_mul]
            sub_len = subquery.shape[0]
            for t in range(sub_len):
                lab = labels[t * self.frame_shift_mul + shift]
                lab = lab[lab != -1]
                song_id = np.searchsorted(self.song_pos, lab, side='right') - 1
                song_t = lab - self.song_pos[song_id] - t
                candidates.append(np.stack([song_id, song_t], axis=1))
            # according to NumPy, np.unique returns sorted array
            candidates = np.unique(np.concatenate(candidates), axis=0)
            
            vec = np.zeros_like(subquery)
            for c in candidates:
                song_id = c[0].item()
                song_start = self.song_pos[song_id].item()
                song_len = self.song_pos[song_id+1].item() - song_start
                t = c[1].item()
                real_time = (t - shift / self.frame_shift_mul) * self.hop_size
                
                # get corresponding embeddings from db
                for i in range(sub_len):
                    if t+i < 0 or t+i >= song_len:
                        vec[i] = 0.0
                    else:
                        self.index.reconstruct(song_start + t+i, vec[i])
                # compute average score
                sco = np.dot(vec.flatten(), subquery.flatten()).item() / sub_len
                if sco > song_score[song_id, 0]:
                    song_score[song_id, 0] = sco
                    song_score[song_id, 1] = real_time
                if sco > best:
                    best = sco
                    best_song_t = song_id, real_time
        tm_3 = time.time()
        logger.info('search %.6fs rerank %.6fs', tm_2-tm_1, tm_3-tm_2)
        return best, best_song_t, song_score

    def query_embeddings_cpp(self, query):
        logger = mp.get_logger()
        tm_1 = time.time()
        d = self.index.d
        distances, labels = self.gpu_index.search(query, self.top_k)
        tm_2 = time.time()
        best = -1e999
        best_song_t = -1, 0
        song_score = np.zeros([self.song_pos.shape[0] - 1, 2], dtype=np.float32)

        song_id = mydll.seq_score(
            int(self.index.this),
            self.song_pos.ctypes.data_as(POINTER(c_int64)),
            self.song_pos.shape[0]-1,
            query.ctypes.data_as(POINTER(c_float)),
            query.shape[0],
            labels.ctypes.data_as(POINTER(c_int64)),
            self.top_k,
            song_score.ctypes.data_as(POINTER(c_float)),
            self.frame_shift_mul,
            self.score_alpha
        )
        best = song_score[song_id, 0].item()
        best_song_t = song_id, song_score[song_id, 1].item() * self.hop_size / self.frame_shift_mul
        tm_3 = time.time()
        song_score[:, 1] *= self.hop_size / self.frame_shift_mul
        logger.info('search %.6fs rerank %.6fs', tm_2-tm_1, tm_3-tm_2)
        return best, best_song_t, song_score
