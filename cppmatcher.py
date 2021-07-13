# need conda to run this program
import csv
import math
import os
import sys
import warnings

import faiss
import julius
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.multiprocessing as mp
import tqdm
import subprocess
if os.name == 'nt':
    print(os.name)
    import msvcrt

# torchaudio currently (0.7) will throw warning that cannot be disabled
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import torchaudio

import simpleutils
from model import FpNetwork
from datautil.musicdata import MusicDataset

if __name__ == "__main__":
    mp.set_start_method('spawn')
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
    with open(os.path.join(dir_for_db, 'songList.txt'), 'r', encoding='utf8') as fin:
        songList = []
        for line in fin:
            if line.endswith('\n'): line = line[:-1]
            songList.append(line)
    
    # doing inference, turn off gradient
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    dataset = MusicDataset(file_list_for_query, params)
    # no task parallelism
    loader = DataLoader(dataset, num_workers=0)

    # open my c++ program
    env = {**os.environ}
    env['LD_LIBRARY_PATH'] = os.environ['CONDA_PREFIX'] + '/lib'
    query_proc = subprocess.Popen(['cpp/faisscputest', dir_for_db]
        , stdin=subprocess.PIPE, stdout=subprocess.PIPE
        , universal_newlines=False, env=env)
    if os.name == 'nt':
        # only Windows needs this!
        print('nt')
        msvcrt.setmode(query_proc.stdin.fileno(), os.O_BINARY)
        msvcrt.setmode(query_proc.stdout.fileno(), os.O_BINARY)
    
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=params['sample_rate'],
        n_fft=params['stft_n'],
        hop_length=params['stft_hop'],
        f_min=params['f_min'],
        f_max=params['f_max'],
        n_mels=params['n_mels'],
        window_fn=torch.hann_window).to(device)
    
    fout = open(result_file, 'w', encoding='utf8', newline='\n')
    fout2 = open(result_file2, 'w', encoding='utf8', newline='\n')
    fout_score = open(result_file_score, 'wb')
    detail_writer = csv.writer(fout2)
    detail_writer.writerow(['query', 'answer', 'score', 'time', 'part_scores'])
    
    torch.set_num_threads(1)
    for dat in tqdm.tqdm(loader):
        embeddings = []
        grads = []
        specs = []
        i, name, wav = dat
        i = int(i) # i is leaking file handles!
        # batch size should be less than 20 because query contains at most 19 segments
        for batch in torch.split(wav.squeeze(0), 16):
            g = batch.to(device)
            
            # Mel spectrogram
            with warnings.catch_warnings():
                # torchaudio is still using deprecated function torch.rfft
                warnings.simplefilter("ignore")
                g = mel(g)
            g = torch.log(g + 1e-12)
            if params.get('spec_norm', 'l2') == 'max':
                g -= torch.amax(g, dim=(1,2)).reshape(-1, 1, 1)
            z = model.forward(g, norm=False).cpu()
            z = torch.nn.functional.normalize(z, p=2)
            embeddings.append(z)
        embeddings = torch.cat(embeddings)
        song_score = np.zeros(len(songList), dtype=np.float32)

        query_proc.stdin.write(embeddings[0::frame_shift_mul].numpy().size.to_bytes(4, 'little'))
        query_proc.stdin.write(embeddings[0::frame_shift_mul].numpy().tobytes())
        query_proc.stdin.flush()
        ans = int.from_bytes(query_proc.stdout.read(4), 'little')

        ans = songList[ans]
        sco = 0
        tim = 0
        upsco = []
        tim /= frame_shift_mul
        tim *= params['hop_size']
        fout.write('%s\t%s\n' % (name[0], ans))
        fout.flush()
        detail_writer.writerow([name[0], ans, sco, tim] + upsco)
        fout2.flush()
        
        #fout_score.write(song_score.tobytes())
    fout.close()
    fout2.close()
else:
    torch.set_num_threads(1)
