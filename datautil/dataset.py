import argparse
import csv
from pathlib import Path
import time
import os
import warnings

import tqdm
from simpleutils import get_hash

import torch
import torch.multiprocessing as mp
import numpy as np
import miniaudio
# torchaudio currently (0.7) will throw warning that cannot be disabled
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import torchaudio
from datautil.audio import get_audio
torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, path, cache_dir='caches', hop_size=0.5, clip_size=1.2, sample_rate=44100):
        super(MyDataset, self).__init__()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.clip_size = 1.2
        self.sample_rate = sample_rate
        with open(path, 'r', encoding='utf8') as fin:
            reader = csv.DictReader(fin)
            files = list(reader)
        self.files = []
        for file in files:
            name = file['file']
            hash = get_hash(name)
            du = float(file['duration'])
            t = 0
            # remove "lock" files
            if (self.cache_dir / (hash+'_')).exists():
                (self.cache_dir / (hash+'_')).unlink()
            while t + clip_size <= du:
                self.files.append({
                    'file': name,
                    'duration': du,
                    'start': t,
                    'hash': hash
                })
                t += hop_size
    
    def load_from_cache(self, hash, frame_offset, num_frames):
        with open(self.cache_dir / hash, 'rb') as fin:
            fin.seek(frame_offset * 2)
            code = fin.read(num_frames * 2)
        # int16 to float32
        wave = np.frombuffer(code, dtype=np.int16).astype(np.float32)
        wave /= 32768
        return torch.FloatTensor(wave.reshape([1, -1]))
    
    def load_from_hdd(self, name, hash):
        wave, smpRate = get_audio(name)
        wave = torch.FloatTensor(wave)
        # stereo to mono
        wave = wave.mean(axis=0, keepdim=True)
        # resample to 44100
        torch.set_num_threads(1)
        wave = torchaudio.compliance.kaldi.resample_waveform(wave, smpRate, self.sample_rate)
        saves = wave.numpy()
        # float32 to int16
        saves = np.clip(saves * 32768, -32768, 32767).astype(np.int16)
        # save to temporary location
        tmppath = self.cache_dir / ('_'+str(os.getpid()))
        saves.tofile(tmppath)
        # save to cache
        try:
            tmppath.rename(self.cache_dir / hash)
        except FileExistsError: # can only happen on Windows
            tmppath.unlink()
        except PermissionError: # can only happen on Windows
            pass
        return torch.FloatTensor(saves) / 32768
    
    def __getitem__(self, index):
        #print('I am %d and I have %d' % (os.getpid(), index))
        file = self.files[index]
        hash = file['hash']
        t_start = int(file['start'] * self.sample_rate)
        t_duration = int(self.clip_size * self.sample_rate)
        lock_file = self.cache_dir / (hash+'_')
        # wait for other workers to finish
        if lock_file.exists():
            while not (self.cache_dir / hash).exists():
                time.sleep(0.01)
        if (self.cache_dir / hash).exists():
            # cache hit!
            return self.load_from_cache(hash, t_start, t_duration)
        # create "lock" file
        with open(lock_file, 'wb'):
            pass
        
        name = file['file']
        wave = self.load_from_hdd(name, hash)
        try:
            lock_file.unlink()
        except (PermissionError, FileNotFoundError):
            pass
        return wave[:, t_start:t_start+t_duration]
    def __len__(self):
        return len(self.files)

def preloader_func(dir, que):
    while True:
        file = que.get()
        if file == '':
            break
        if (dir/file).exists():
            with open(dir / file, 'rb') as fin:
                preloaded = fin.read()

class MySampler(torch.utils.data.Sampler):
    def __init__(self, data_source, chunk_size):
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.generator = torch.Generator()
        self.generator2 = torch.Generator()
    def __iter__(self):
        que = mp.Manager().Queue()
        preloader = mp.Process(target=preloader_func, args=(self.data_source.cache_dir, que,))
        n = len(self.data_source)
        bs = self.chunk_size
        n_chunks = (n-1) // bs + 1
        preloader.start()
        shuffled = torch.randperm(n_chunks, generator=self.generator).tolist()
        # send file names to preloader
        which = set()
        i = shuffled[0]
        for j in range(i*bs, min((i+1)*bs, n)):
            file = self.data_source.files[j]
            if file['hash'] not in which:
                which.add(file['hash'])
                que.put(file['hash'])
        for i in range(len(shuffled)):
            start = shuffled[i] * bs
            size = min(bs, n - start)
            if i+1 < len(shuffled):
                # send file names to preloader
                for j in range(shuffled[i+1]*bs, min((shuffled[i+1]+1)*bs, n)):
                    file = self.data_source.files[j]
                    if file['hash'] not in which:
                        which.add(file['hash'])
                        que.put(file['hash'])
            yield from (start + torch.randperm(size, generator=self.generator2)).tolist()
        # stop preloader
        que.put('')
        preloader.join()
    def __len__(self):
        return len(self.data_source)

if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('--csv', required=True)
    argp.add_argument('--cache-dir', default='caches')
    argp.add_argument('--workers', type=int, default=0)
    args = argp.parse_args()
    
    mp.set_start_method('spawn')
    dataset = MyDataset(args.csv, cache_dir=args.cache_dir)
    sampler = MySampler(dataset, 20000)
    loader = torch.utils.data.DataLoader(dataset, num_workers=args.workers, sampler=sampler, batch_size=320)
    
    print('test dataloader for training data')
    for epoch in range(3):
        for i in tqdm.tqdm(loader):
            pass
