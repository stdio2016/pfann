import csv
import math
import os
import sys
import time
import warnings

import faiss
import numpy as np
import tqdm
import torch.multiprocessing as mp

import simpleutils
from database import Database

if __name__ == "__main__":
    logger_init = simpleutils.MultiProcessInitLogger('matchemb')
    logger_init()
    
    mp.set_start_method('spawn')
    logger = mp.get_logger()
    if len(sys.argv) < 4:
        print('Usage: python %s <query embedding dir> <database dir> <result file>' % sys.argv[0])
        sys.exit()
    dir_for_query = sys.argv[1]
    dir_for_db = sys.argv[2]
    result_file = sys.argv[3]
    result_file2 = os.path.splitext(result_file) # for more detailed output
    result_file2 = result_file2[0] + '_detail.csv'
    result_file_score = result_file + '.bin'
    configs = os.path.join(dir_for_db, 'configs.json')
    params = simpleutils.read_config(configs)
    file_list = simpleutils.read_file_list(os.path.join(dir_for_query, 'queryList.txt'))
    logger.info('command args: %s', sys.argv)
    logger.info('params: %s', params)

    d = params['model']['d']

    top_k = params['indexer']['top_k']
    frame_shift_mul = params['indexer'].get('frame_shift_mul', 1)

    print('loading database...')
    db = Database(dir_for_db, params['indexer'], params['hop_size'])
    print('database loaded')

    print('loading queries')
    query_embeddings = np.fromfile(os.path.join(dir_for_query, 'query_embeddings'), dtype=np.float32)
    query_embeddings = query_embeddings.reshape([-1, d])
    query_index = np.fromfile(os.path.join(dir_for_query, 'query_index'), dtype=np.int64)
    query_index = query_index.reshape([-1, 2])
    print('queries loaded')

    tm_0 = time.time()
    fout = open(result_file, 'w', encoding='utf8', newline='\n')
    fout2 = open(result_file2, 'w', encoding='utf8', newline='\n')
    fout_score = open(result_file_score, 'wb')
    detail_writer = csv.writer(fout2)
    detail_writer.writerow(['query', 'answer', 'score', 'time', 'part_scores'])
    for i, name in enumerate(tqdm.tqdm(file_list)):
        logger.info('get query %s', name)
        tm_1 = time.time()
        
        my_idx = query_index[i, 0]
        my_len = query_index[i, 1]
        embeddings = query_embeddings[my_idx:my_idx+my_len]

        tm_2 = time.time()
        logger.info('compute embedding %.6fs', tm_2 - tm_1)

        sco, (ans, tim), song_score = db.query_embeddings(embeddings)
        upsco = []
        ans = db.songList[ans]

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
