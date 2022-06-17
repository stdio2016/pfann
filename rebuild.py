# rebuild is for reindexing embedding database using different index
import os
import shutil
import sys
import time
import warnings

import faiss
import numpy as np

import simpleutils

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python %s <db location>' % sys.argv[0])
        sys.exit()
    dir_for_db = sys.argv[1]
    configs = os.path.join(dir_for_db, 'configs.json')
    params = simpleutils.read_config(configs)

    d = params['model']['d']
    h = params['model']['h']
    u = params['model']['u']
    F_bin = params['n_mels']
    segn = int(params['segment_size'] * params['sample_rate'])
    T = (segn + params['stft_hop'] - 1) // params['stft_hop']

    print('loading embeddings')
    embeddings = np.fromfile(os.path.join(dir_for_db, 'embeddings'), dtype=np.float32).reshape([-1, d])

    # train indexer
    print('training indexer')
    index = faiss.index_factory(d, params['indexer']['index_factory'], faiss.METRIC_INNER_PRODUCT)
    index.verbose = True
    
    if not index.is_trained:
        try:
            index.train(embeddings)
        except RuntimeError as x:
            print(x)
            if "Error: 'nx >= k' failed" in str(x):
                index = faiss.IndexFlatIP(d)
    #index = faiss.IndexFlatIP(d)
    
    # write database
    print('writing database')
    index.add(embeddings)
    faiss.write_index(index, os.path.join(dir_for_db, 'landmarkValue'))
