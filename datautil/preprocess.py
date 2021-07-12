import csv
import os

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
import torchaudio
import tqdm

from datautil.audio import get_audio

class Preprocessor(Dataset):
    def __init__(self, files, dir, sample_rate):
        self.files = files
        self.dir = dir
        self.resampler = {}
        self.sample_rate = sample_rate

    def __getitem__(self, n):
        dat = get_audio(os.path.join(self.dir, self.files[n]))
        wav, smprate = dat
        if smprate not in self.resampler:
            self.resampler[smprate] = torchaudio.transforms.Resample(smprate, self.sample_rate)
        wav = torch.Tensor(wav)
        wav = wav.mean(dim=0)
        wav = self.resampler[smprate](torch.Tensor(wav))

        # quantize to 16 bit again
        wav *= 32768
        torch.clamp(wav, -32768, 32767, out=wav)
        wav = wav.to(torch.int16)
        return wav

    def __len__(self):
        return len(self.files)

def preprocess_music(music_dir, music_csv, sample_rate, preprocess_out):
    print('converting music to wav')
    with open(music_csv) as fin:
        reader = csv.reader(fin)
        next(reader)
        files = [row[0] for row in reader]

    preprocessor = Preprocessor(files, music_dir, sample_rate)
    loader = DataLoader(preprocessor, num_workers=4, batch_size=None)
    out_file = open(preprocess_out + '.bin', 'wb')
    song_lens = []
    for wav in tqdm.tqdm(loader):
        # torch.set_num_threads(1) # default multithreading causes cpu contention
        
        wav = wav.numpy()
        out_file.write(wav.tobytes())
        song_lens.append(wav.shape[0])
    out_file.close()
    np.save(preprocess_out, np.array(song_lens, dtype=np.int64))
