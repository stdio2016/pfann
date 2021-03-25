import argparse
import csv
import os
import warnings

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
    def __init__(self, music_dir, music_list, noise, air, micirp, query_len, params):
        self.music_dir = music_dir
        self.music_list = music_list
        self.noise = noise
        self.air = air
        self.micirp = micirp
        self.query_len = query_len
        self.params = params
        self.pad_start = params['pad_start']
        self.sample_rate = params['sample_rate']
    
    def __getitem__(self, index):
        # load music
        name = self.music_list[index % len(self.music_list)]
        music, smprate = get_audio(os.path.join(self.music_dir, name))
        
        # crop a music clip
        sel_smp = int(smprate * self.query_len)
        pad_smp = int(smprate * self.pad_start)
        hop_smp = int(smprate * self.params['hop_size'])
        if music.shape[1] > sel_smp:
            time_offset = torch.randint(low=0, high=music.shape[1]-sel_smp, size=(1,))
            music = music[:, max(0,time_offset-pad_smp):time_offset+sel_smp]
            music = np.pad(music, ((0,0), (max(pad_smp-time_offset,0),0)))
        else:
            time_offset = 0
            music = np.pad(music, ((0,0), (pad_smp, sel_smp-music.shape[1])))
        music = torch.from_numpy(music)
        
        # stereo to mono and resample
        music = music.mean(dim=0)
        music = torchaudio.transforms.Resample(smprate, self.sample_rate)(music)
        
        # fix size
        sel_smp = int(self.sample_rate * self.query_len)
        pad_smp = int(self.sample_rate * self.pad_start)
        if music.shape[0] > sel_smp+pad_smp:
            music = music[:sel_smp+pad_smp]
        else:
            music = F.pad(music, (0, sel_smp+pad_smp-music.shape[0]))
        
        # background mixing
        music -= music.mean()
        amp = torch.sqrt((music**2).mean())
        snr_max = self.params['noise']['snr_max']
        snr_min = self.params['noise']['snr_min']
        snr = snr_min + torch.rand(1) * (snr_max - snr_min)
        if self.noise:
            noise = self.noise.random_choose(1, music.shape[0])[0]
            noise_amp = torch.sqrt((noise**2).mean())
            noise = noise * (amp / noise_amp * torch.pow(10, -0.05*snr))
        else:
            noise = torch.normal(mean=torch.zeros_like(music), std=(amp*torch.pow(10, -0.05*snr)))
        
        # IR filters
        music_freq = torch.fft.rfft(music, self.params['fftconv_n'])
        noise_freq = torch.fft.rfft(noise, self.params['fftconv_n'])
        if self.air:
            aira, reverb = self.air.random_choose_name()
            music_freq *= aira
            noise_freq *= aira
        if self.micirp:
            micirp = self.micirp.random_choose(1)[0]
            music_freq *= micirp
            noise_freq *= micirp
        music = torch.fft.irfft(music_freq, self.params['fftconv_n'])
        music = music[pad_smp:pad_smp+sel_smp]
        noise = torch.fft.irfft(noise_freq, self.params['fftconv_n'])
        noise = noise[pad_smp:pad_smp+sel_smp]
        mix = music + noise
        
        # normalize volume
        vol = max(torch.max(torch.abs(mix)), torch.max(torch.abs(music)), torch.max(torch.abs(noise)))
        music /= vol
        noise /= vol
        mix /= vol
        
        return name, music, noise, mix
    
    def __len__(self):
        return len(self.music_list)

def gen_for(train_val, args, params):
    sample_rate = params['sample_rate']
    
    if args.noise:
        noise = NoiseData(noise_dir=args.noise,
            list_csv=params['noise'][train_val],
            sample_rate=sample_rate, cache_dir=params['cache_dir'])
    else:
        noise = None
    
    if args.air:
        air = AIR(air_dir=args.air,
            list_csv=params['air'][train_val],
            length=params['air']['length'],
            fftconv_n=params['fftconv_n'], sample_rate=sample_rate)
    else:
        air = None
    
    if args.micirp:
        micirp = MicIRP(mic_dir=args.micirp,
            list_csv=params['micirp'][train_val],
            length=params['micirp']['length'],
            fftconv_n=params['fftconv_n'], sample_rate=sample_rate)
    else:
        micirp = None
    
    with open(params[train_val+'_csv'], 'r') as fin:
        music_list = []
        reader = csv.reader(fin)
        next(reader)
        for line in reader:
            music_list.append(line[0])
    
    gen = QueryGen(args.data, music_list, noise, air, micirp, args.length, params)
    runall = torch.utils.data.DataLoader(
        dataset=gen,
        num_workers=3
    )
    os.makedirs(args.out, exist_ok=True)
    fout = open(os.path.join(args.out, 'denoise_'+train_val+'.csv'), 'w', encoding='utf8', newline='\n')
    writer = csv.writer(fout)
    writer.writerow(['mix_path', 'music_path', 'noise_path'])
    os.makedirs(os.path.join(args.out, 'music'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'mix'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'noise'), exist_ok=True)
    for i, (name,music,noise,mix) in enumerate(tqdm.tqdm(runall)):
        name = os.path.split(name[0])[1]
        name = os.path.splitext(name)[0] + '.wav'
        writer.writerow(['music/'+name, 'mix/'+name, 'noise/'+name])
        
        torchaudio.save(os.path.join(args.out, 'music', name), music, gen.sample_rate)
        torchaudio.save(os.path.join(args.out, 'mix', name), mix, gen.sample_rate)
        torchaudio.save(os.path.join(args.out, 'noise', name), noise, gen.sample_rate)
    fout.close()

if __name__ == '__main__':
    # don't delete this line, because my data loader uses queues
    torch.multiprocessing.set_start_method('spawn')
    args = argparse.ArgumentParser()
    args.add_argument('-d', '--data', required=True)
    args.add_argument('--noise')
    args.add_argument('--air')
    args.add_argument('--micirp')
    args.add_argument('-p', '--params', default='configs/default.json')
    args.add_argument('-l', '--length', type=float, default=30)
    args.add_argument('-o', '--out', required=True)
    args = args.parse_args()
    
    params = simpleutils.read_config(args.params)
    sample_rate = params['sample_rate']
    win = (params['pad_start'] + args.length + params['air']['length'] + params['micirp']['length']) * sample_rate
    train_val = 'validate'
    fftconv_n = 2048
    while fftconv_n < win:
        fftconv_n *= 2
    params['fftconv_n'] = fftconv_n
    gen_for('train', args, params)
    gen_for('validate', args, params)
