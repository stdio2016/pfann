import argparse
import csv
import os
import warnings
import json

import numpy as np
import torch
import torch.nn.functional as F
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import torchaudio
import tqdm
import scipy.signal

import simpleutils
from datautil.audio import get_audio
from datautil.ir import AIR, MicIRP
from datautil.noise import NoiseData

def biquad_faster(waveform, b0, b1, b2, a0, a1, a2):
    waveform = waveform.numpy()
    b = np.array([b0, b1, b2], dtype=waveform.dtype)
    a = np.array([a0, a1, a2], dtype=waveform.dtype)
    return torch.from_numpy(scipy.signal.lfilter(b, a, waveform))
torchaudio.functional.biquad = biquad_faster

class QueryGen(torch.utils.data.Dataset):
    def __init__(self, music_dir, music_list, noise, air, micirp, query_len, num_queries, params):
        self.music_dir = music_dir
        self.music_list = music_list
        self.noise = noise
        self.air = air
        self.micirp = micirp
        self.query_len = query_len
        self.num_queries = num_queries
        self.params = params
        self.pad_start = params['pad_start']
        self.sample_rate = params['sample_rate']
    
    def __getitem__(self, index):
        torch.manual_seed(9000 + index)
        # load music
        name = self.music_list[index % len(self.music_list)]
        audio, smprate = get_audio(os.path.join(self.music_dir, name))
        audio = torch.from_numpy(audio.astype(np.float32))
        
        # stereo to mono and resample
        audio = audio.mean(dim=0)
        audio = torchaudio.transforms.Resample(smprate, self.sample_rate)(audio)
        
        # random crop
        sel_smp = int(self.sample_rate * self.params['segment_size'])
        total_segs = max(int(audio.shape[0] / sel_smp), 1)
        shift_smp = int(self.sample_rate * self.params['time_offset']) - sel_smp
        crop_pos = torch.randint(low=-shift_smp, high=shift_smp+1, size=[total_segs])
        segs = []
        for i in range(total_segs):
            offset = crop_pos[i] + sel_smp * i
            seg = audio[max(0,offset) : max(0,offset+sel_smp)]
            seg = F.pad(seg, (
                max(0, -offset),
                max(0, (offset+sel_smp)-audio.shape[0])
            ))
            segs.append(seg)
        audio = torch.stack(segs, 0)
        
        # background mixing
        audio -= audio.mean(1, keepdim=True)
        snr_max = self.params['noise']['snr_max']
        snr_min = self.params['noise']['snr_min']
        audio = self.noise.add_noises(audio, snr_min, snr_max)
        
        # IR filters
        audio_freq = torch.fft.rfft(audio, self.params['fftconv_n'])
        if self.air:
            audio_freq *= self.air.random_choose(audio_freq.shape[0])
        if self.micirp:
            audio_freq *= self.micirp.random_choose(audio_freq.shape[0])
        audio = torch.fft.irfft(audio_freq, self.params['fftconv_n'])
        audio = audio[..., 0:0+sel_smp]
        
        # normalize volume
        audio = F.normalize(audio, p=np.inf, dim=1)
        
        # random select part
        audio = audio.flatten()
        hop_size = int(self.params['hop_size'] * self.sample_rate)
        n_segs = int((audio.shape[0] - sel_smp) / hop_size) + 1
        q_len = int(self.query_len * self.sample_rate)
        need_segs = int((q_len - sel_smp) / hop_size) + 1
        r = torch.randint(0, n_segs - need_segs + 1, (1,)).item()
        time_offset = r * hop_size
        audio = audio[r*hop_size : r*hop_size+q_len]
        
        return name, time_offset / self.sample_rate, audio
    
    def __len__(self):
        return self.num_queries

if __name__ == '__main__':
    # don't delete this line, because my data loader uses queues
    torch.multiprocessing.set_start_method('spawn')
    args = argparse.ArgumentParser()
    args.add_argument('-p', '--params', default='configs/default.json')
    args.add_argument('-l', '--length', type=float, default=1)
    args.add_argument('--num', type=int, default=10)
    args.add_argument('--mode', default='test', choices=['train', 'validate', 'test'])
    args.add_argument('-o', '--out', required=True)
    args = args.parse_args()
    
    # warn user (actually just me!) if query files exist
    if os.path.exists(args.out):
        yesno = input('Folder %s exists, overwrite anyway? (y/n) ' % args.out)
        while yesno not in {'y', 'n'}:
            yesno = input('Please enter y or n: ')
        if yesno == 'n':
            exit()
    
    params = simpleutils.read_config(args.params)
    train_val = 'validate' if args.mode == 'test' else args.mode
    train_val_test = args.mode
    sample_rate = params['sample_rate']
    win = (params['pad_start'] + args.length + params['air']['length'] + params['micirp']['length']) * sample_rate
    fftconv_n = 2048
    while fftconv_n < win:
        fftconv_n *= 2
    params['fftconv_n'] = fftconv_n
    
    noise = NoiseData(noise_dir=params['noise']['dir'],
            list_csv=params['noise'][train_val],
            sample_rate=sample_rate, cache_dir=params['cache_dir'])
    
    air = AIR(air_dir=params['air']['dir'],
            list_csv=params['air'][train_val],
            length=params['air']['length'],
            fftconv_n=params['fftconv_n'], sample_rate=sample_rate)
    
    micirp = MicIRP(mic_dir=params['micirp']['dir'],
            list_csv=params['micirp'][train_val],
            length=params['micirp']['length'],
            fftconv_n=params['fftconv_n'], sample_rate=sample_rate)
    
    music_list = simpleutils.read_file_list(params[train_val_test + '_csv'])
    
    gen = QueryGen(params['music_dir'], music_list, noise, air, micirp, args.length, args.num, params)
    runall = torch.utils.data.DataLoader(
        dataset=gen,
        num_workers=3,
        batch_size=None
    )
    os.makedirs(args.out, exist_ok=True)
    fout = open(os.path.join(args.out, 'expected.csv'), 'w', encoding='utf8', newline='\n')
    fout2 = open(os.path.join(args.out, 'list.txt'), 'w', encoding='utf8')
    writer = csv.writer(fout)
    writer.writerow(['query', 'answer', 'time'])
    for i, (name,time_offset,sound) in enumerate(tqdm.tqdm(runall)):
        safe_name = os.path.splitext(os.path.split(name)[1])[0]
        out_name = 'q%04d_%s_%.1f.wav' % (i+1, safe_name, time_offset)
        writer.writerow([out_name, name, time_offset])
        path = os.path.join(args.out, out_name)
        torchaudio.save(path, sound.unsqueeze(0), gen.sample_rate, encoding='PCM_S', bits_per_sample=16)
        fout2.write(path + '\n')
    fout.close()
    fout2.close()
    params['genquery'] = {'mode': train_val_test, 'length': args.length}
    with open(os.path.join(args.out, 'configs.json'), 'w') as fout:
        json.dump(params, fout, indent=2)
