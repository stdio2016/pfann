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
        # load music
        name = self.music_list[index % len(self.music_list)]
        audio, smprate = get_audio(os.path.join(self.music_dir, name))
        
        # crop a music clip
        sel_smp = int(smprate * self.query_len)
        pad_smp = int(smprate * self.pad_start)
        hop_smp = int(smprate * self.params['hop_size'])
        if audio.shape[1] >= sel_smp:
            time_offset = torch.randint(low=0, high=audio.shape[1]-sel_smp, size=(1,))
            audio = audio[:, max(0,time_offset-pad_smp):time_offset+sel_smp]
            audio = np.pad(audio, ((0,0), (max(pad_smp-time_offset,0),0)))
        else:
            time_offset = 0
            audio = np.pad(audio, ((0,0), (pad_smp, sel_smp-audio.shape[1])))
        audio = torch.from_numpy(audio)
        
        # stereo to mono and resample
        audio = audio.mean(dim=0)
        audio = torchaudio.transforms.Resample(smprate, self.sample_rate)(audio)
        
        # fix size
        sel_smp = int(self.sample_rate * self.query_len)
        pad_smp = int(self.sample_rate * self.pad_start)
        if audio.shape[0] > sel_smp+pad_smp:
            audio = audio[:sel_smp+pad_smp]
        else:
            audio = F.pad(audio, (0, sel_smp+pad_smp-audio.shape[0]))
        
        # background mixing
        audio -= audio.mean()
        # our model cannot hear <300Hz sound
        if self.params['noise'].get('snr_only_in_f_range', False):
            audio_hi = torchaudio.functional.bass_biquad(audio, self.sample_rate, -24, self.params['f_min'])
            amp = torch.sqrt((audio_hi**2).mean())
        else:
            amp = torch.sqrt((audio**2).mean())
        snr_max = self.params['noise']['snr_max']
        snr_min = self.params['noise']['snr_min']
        snr = snr_min + torch.rand(1) * (snr_max - snr_min)
        if self.noise:
            noise = self.noise.random_choose(1, audio.shape[0])[0]
            # our model cannot hear <300Hz sound
            if self.params['noise'].get('snr_only_in_f_range', False):
                noise_hi = torchaudio.functional.bass_biquad(noise, self.sample_rate, -24, self.params['f_min'])
                noise_amp = torch.sqrt((noise_hi**2).mean())
            else:
                noise_amp = torch.sqrt((noise**2).mean())
            audio += noise * (amp / noise_amp * torch.pow(10, -0.05*snr))
        else:
            audio = torch.normal(mean=audio, std=(amp*torch.pow(10, -0.05*snr)))
        
        # IR filters
        audio_freq = torch.fft.rfft(audio, self.params['fftconv_n'])
        reverb = ''
        if self.air:
            aira, reverb = self.air.random_choose_name()
            audio_freq *= aira
        if self.micirp:
            audio_freq *= self.micirp.random_choose(1)[0]
        audio = torch.fft.irfft(audio_freq, self.params['fftconv_n'])
        audio = audio[pad_smp:pad_smp+sel_smp]
        
        # normalize volume
        #audio = F.normalize(audio, p=np.inf, dim=0)
        audio = audio / 3
        audio = torch.clip(audio, -1, 1)
        
        return name, time_offset/smprate, audio, snr, reverb
    
    def __len__(self):
        return self.num_queries

if __name__ == '__main__':
    # don't delete this line, because my data loader uses queues
    torch.multiprocessing.set_start_method('spawn')
    args = argparse.ArgumentParser()
    args.add_argument('-d', '--data', required=True)
    args.add_argument('--noise')
    args.add_argument('--air')
    args.add_argument('--micirp')
    args.add_argument('-p', '--params', default='configs/default.json')
    args.add_argument('-l', '--length', type=float, default=1)
    args.add_argument('--num', type=int, default=10)
    args.add_argument('-o', '--out', required=True)
    args = args.parse_args()
    
    params = simpleutils.read_config(args.params)
    train_val = 'validate'
    sample_rate = params['sample_rate']
    win = (params['pad_start'] + args.length + params['air']['length'] + params['micirp']['length']) * sample_rate
    fftconv_n = 2048
    while fftconv_n < win:
        fftconv_n *= 2
    params['fftconv_n'] = fftconv_n
    
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
    
    with open(params['test_csv'], 'r') as fin:
        music_list = []
        reader = csv.reader(fin)
        next(reader)
        for line in reader:
            music_list.append(line[0])
    
    gen = QueryGen(args.data, music_list, noise, air, micirp, args.length, args.num, params)
    runall = torch.utils.data.DataLoader(
        dataset=gen,
        num_workers=3
    )
    os.makedirs(args.out, exist_ok=True)
    fout = open(os.path.join(args.out, 'expected.csv'), 'w', encoding='utf8', newline='\n')
    fout2 = open(os.path.join(args.out, 'list.txt'), 'w', encoding='utf8')
    writer = csv.writer(fout)
    writer.writerow(['query', 'answer', 'time', 'snr', 'reverb'])
    for i, (name,time_offset,sound,snr,reverb) in enumerate(tqdm.tqdm(runall)):
        writer.writerow(['q%04d.wav' % (i+1), name[0], float(time_offset), float(snr), reverb[0]])
        path = os.path.join(args.out, 'q%04d.wav' % (i+1))
        torchaudio.save(path, sound, gen.sample_rate)
        fout2.write(path + '\n')
    fout.close()
    fout2.close()
