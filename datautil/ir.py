import argparse
import csv
import os
import warnings

import scipy.io
import numpy as np
import torch
import torch.fft
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import torchaudio

from datautil.audio import get_audio

class AIR:
    def __init__(self, air_dir, list_csv, length, fftconv_n, sample_rate=8000):
        print('loading Aachen IR dataset')
        with open(list_csv, 'r') as fin:
            reader = csv.reader(fin)
            airs = []
            firstrow = next(reader)
            for row in reader:
                airs.append(row[0])
        data = []
        to_len = int(length * sample_rate)
        self.names = []
        for name in airs:
            mat = scipy.io.loadmat(os.path.join(air_dir, name))
            h_air = torch.tensor(mat['h_air'].astype(np.float32))
            assert h_air.shape[0] == 1
            h_air = h_air[0]
            air_info = mat['air_info']
            fs = int(air_info['fs'][0][0][0][0])
            self.names.append(str(air_info['room'][0][0][0]))
            resampled = torchaudio.transforms.Resample(fs, sample_rate)(h_air)
            truncated = resampled[0:to_len]
            freqd = torch.fft.rfft(truncated, fftconv_n)
            data.append(freqd)
        self.data = torch.stack(data)
    
    def random_choose(self, num):
        indices = torch.randint(0, self.data.shape[0], size=(num,), dtype=torch.long)
        return self.data[indices]
    
    def random_choose_name(self):
        index = torch.randint(0, self.data.shape[0], size=(1,), dtype=torch.long).item()
        return self.data[index], self.names[index]

class MicIRP:
    def __init__(self, mic_dir, list_csv, length, fftconv_n, sample_rate=8000):
        print('loading microphone IR dataset')
        with open(list_csv, 'r') as fin:
            reader = csv.reader(fin)
            mics = []
            firstrow = next(reader)
            for row in reader:
                mics.append(row[0])
        data = []
        to_len = int(length * sample_rate)
        for name in mics:
            smp, smprate = get_audio(os.path.join(mic_dir, name))
            smp = torch.FloatTensor(smp).mean(dim=0)
            resampled = torchaudio.transforms.Resample(smprate, sample_rate)(smp)
            truncated = resampled[0:to_len]
            freqd = torch.fft.rfft(truncated, fftconv_n)
            data.append(freqd)
        self.data = torch.stack(data)
    
    def random_choose(self, num):
        indices = torch.randint(0, self.data.shape[0], size=(num,), dtype=torch.long)
        return self.data[indices]

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('air')
    args.add_argument('out')
    args = args.parse_args()
    
    with open(args.out, 'w', encoding='utf8', newline='\n') as fout:
        writer = csv.writer(fout)
        writer.writerow(['file'])
        files = []
        for name in os.listdir(args.air):
            if name.endswith('.mat'):
                files.append(name)
        files.sort()
        for name in files:
            writer.writerow([name])
