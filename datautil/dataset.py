import argparse
import csv
from pathlib import Path
import time
import os
import warnings
import struct

import tqdm
from simpleutils import get_hash

import torch
import torch.fft
import torch.multiprocessing as mp
import torch.nn.functional as F
import numpy as np
import miniaudio
import scipy.signal
# torchaudio currently (0.7) will throw warning that cannot be disabled
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import torchaudio
from datautil.audio import get_audio
torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, path, cache_dir='caches', hop_size=0.5, clip_size=1.2, sample_rate=8000, clips_per_song=60, sel_size=1, pad_start=1):
        super(MyDataset, self).__init__()
        self.clip_size = int(clip_size * sample_rate)
        self.sel_size = int(sel_size * sample_rate)
        self.hop_size = int(hop_size * sample_rate)
        self.pad_start = int(pad_start * sample_rate)
        self.sample_rate = sample_rate
        self.clips_per_song = clips_per_song
        with open(path, 'r', encoding='utf8') as fin:
            reader = csv.DictReader(fin)
            self.files = [f['file'] for f in reader]
        self.prepare_cache(cache_dir, self.files)
    
    def prepare_cache(self, cache_dir, files):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.durations = torch.zeros(len(files), dtype=torch.int)
        pool = mp.Pool(4)
        songPos = [None] * len(files)
        nsegs = 0
        for i, usable in tqdm.tqdm(pool.imap_unordered(self.load_from_hdd, enumerate(files)), total=len(files)):
            songPos[i] = usable
            nsegs += len(usable)
        
        # collect songs segmented by 30sec
        partToSong = []
        partPos = []
        n = len(self.files)
        songToPart = np.zeros(n+1, dtype=np.int)
        for i in range(n):
            t = 0
            while t + self.clip_size <= self.durations[i]:
                partToSong.append(i)
                partPos.append(t)
                t += self.clips_per_song * self.hop_size
            songToPart[i+1] = len(partToSong)
        self.partToSong = torch.tensor(partToSong, dtype=torch.int)
        self.partPos = torch.tensor(partPos, dtype=torch.int)
        
        # collect songs segmented by 1sec
        self.segToPart = np.repeat(songToPart[0:-1], [len(x) for x in songPos])
        self.segPos = np.concatenate(songPos)
        self.segToPart += self.segPos // (self.clips_per_song * self.hop_size)
        self.partToSeg = np.concatenate([[0], np.cumsum(np.bincount(self.segToPart))])
        
        self.segToPart = torch.tensor(self.segToPart)
        self.segPos = torch.tensor(self.segPos)
        self.partToSeg = torch.tensor(self.partToSeg)
    
    def load_from_cache(self, hash, frame_offset, num_frames):
        with open(self.cache_dir / hash, 'rb') as fin:
            fin.seek(frame_offset * 2)
            code = fin.read(num_frames * 2)
        # int16 to float32
        wave = np.frombuffer(code, dtype=np.int16).astype(np.float32)
        wave /= 32768
        return torch.FloatTensor(wave.reshape([1, -1]))
    
    def load_from_hdd(self, i_name):
        i, name = i_name
        hash = get_hash(name)
        (self.cache_dir/hash[0:2]).mkdir(exist_ok=True)
        cachepath = self.cache_dir/hash[0:2]/hash[2:]
        if cachepath.exists():
            with open(cachepath, 'rb') as fout:
                duration = np.frombuffer(fout.read(4), dtype=np.int32)[0]
                self.durations[i] = duration
                fout.seek(4 + duration * 2)
                usable = np.frombuffer(fout.read(), dtype=np.int32)
            return i, usable
        wave, smpRate = get_audio(name)
        wave = torch.FloatTensor(wave)
        # stereo to mono
        wave = wave.mean(axis=0, keepdim=True)
        # resample to 44100
        torch.set_num_threads(1)
        wave = torchaudio.compliance.kaldi.resample_waveform(wave, smpRate, self.sample_rate)
        duration = int(wave.shape[1])
        # find usable segments
        t = 0
        vol = float(torch.max(torch.abs(wave)))
        usable = []
        while t + self.clip_size < duration:
            part = wave[:, t:t+self.clip_size]
            if float(torch.max(torch.abs(part))) > vol * 1e-4:
                usable.append(t)
            else:
                print('%s frame %f dropped' % (name, t/self.sample_rate))
            t += self.hop_size
        usable = np.array(usable, dtype=np.int32)
        # float32 to int16
        torch.clamp(wave * 32768, -32768, 32767, out=wave)
        wave = wave.type(torch.int16)
        self.durations[i] = duration
        # save to cache
        wave = wave.numpy().flatten()
        with open(cachepath, 'wb') as fout:
            fout.write(np.array([duration], dtype=np.int32))
            fout.write(wave)
            fout.write(usable)
        return i, usable
    
    def __getitem__(self, index):
        #print('I am %d and I have %d' % (os.getpid(), index))
        wave, pad_start, start, du = index
        wave = wave[start-pad_start:start+du].to(torch.float32)
        wave *= 1/32768
        pos = torch.randint(0, du-self.sel_size, size=(1,))
        wav1 = wave[pad_start:pad_start+self.sel_size]
        wav2 = wave[max(0, pad_start+pos-self.pad_start) : pad_start+self.sel_size+pos]
        wav2 = F.pad(wav2, (self.pad_start+du-len(wav2), 0))
        return (wav1, wav2)
    def __len__(self):
        return len(self.segToPart)

def preloader_func(dir, in_que, out_que, pad_start):
    loadlist = in_que.get()
    n_chunk = 0
    while loadlist is not None:
        # get total size
        total = 0
        for file, start, du in loadlist:
            total += start - max(start - pad_start, 0) + du
        dats = np.zeros(total, dtype=np.int16)
        # load file to a big numpy array
        total = 0
        posinfo = []
        for file, start, du in loadlist:
            prestart = max(start - pad_start, 0)
            paddu = (start-prestart) + du
            with open(dir / file, 'rb') as fin:
                fin.seek(4 + prestart*2)
                preloaded = np.frombuffer(fin.read(paddu*2), dtype=np.int16)
            #print(paddu, len(preloaded))
            dats[total:total+paddu] = preloaded
            posinfo.append({'padStart':start-prestart, 'start':total+(start-prestart), 'duration':du})
            total += paddu
        out_que.put((torch.tensor(dats), posinfo))
        loadlist = in_que.get()
        n_chunk += 1

class MySampler(torch.utils.data.Sampler):
    def __init__(self, data_source, chunk_size):
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.generator = torch.Generator()
        self.generator2 = torch.Generator()
        self.pad_start = data_source.pad_start
    
    def _preload(self, shuffled, start, que):
        dataset = self.data_source
        cnt = 0
        end = start
        while end < len(shuffled) and cnt < self.chunk_size:
            idx = shuffled[end]
            cnt += int(dataset.partToSeg[idx+1] - dataset.partToSeg[idx])
            end += 1
        loadlist = []
        for partId in shuffled[start:end]:
            file = dataset.files[dataset.partToSong[partId]]
            hash = get_hash(file)
            hash = hash[0:2]+'/'+hash[2:]
            total = int(dataset.durations[dataset.partToSong[partId]])
            startTime = int(dataset.partPos[partId])
            duration = min(total-startTime, (dataset.clips_per_song-1)*dataset.hop_size + dataset.clip_size)
            loadlist.append((hash, startTime, duration))
        que.put(loadlist)
        return end, cnt
    
    def __iter__(self):
        in_que = mp.Queue()
        out_que = mp.Queue()
        preloader = mp.Process(target=preloader_func, args=(self.data_source.cache_dir, in_que, out_que, self.pad_start))
        dataset = self.data_source
        n = len(dataset.partToSong)
        preloader.start()
        shuffled = torch.randperm(n, generator=self.generator).tolist()
        chunkpos = 0
        next_chunkpos, cnt = self._preload(shuffled, chunkpos, in_que)
        while chunkpos < n:
            if next_chunkpos < n:
                # send file names to preloader
                next_chunkpos2, cnt2 = self._preload(shuffled, next_chunkpos, in_que)
            # shuffle current chunk
            segIds = []
            for partId in shuffled[chunkpos : next_chunkpos]:
                segIds.append(torch.arange(dataset.partToSeg[partId], dataset.partToSeg[partId+1]))
            segIds = torch.cat(segIds)
            loaded, posinfo = out_que.get()
            posdict = {}
            for i, info in enumerate(posinfo):
                partId = shuffled[chunkpos+i]
                posdict[partId] = info
            for i in torch.randperm(len(segIds), generator=self.generator2).tolist():
                # look up for segment position
                segId = int(segIds[i])
                partId = int(dataset.segToPart[segId])
                partPos = int(dataset.partPos[partId])
                segPos = int(dataset.segPos[segId])
                padInfo = posdict[partId]
                seg_start = padInfo['start'] + (segPos - partPos)
                seg_padStart = min(self.pad_start, padInfo['padStart'])
                seg_du = dataset.clip_size
                yield (loaded, seg_padStart, seg_start, seg_du)
            chunkpos = next_chunkpos
            if next_chunkpos < n:
                next_chunkpos = next_chunkpos2
                cnt = cnt2
        # stop preloader
        in_que.put(None)
        preloader.join()
    
    def __len__(self):
        return len(self.data_source)
    
    def set_epoch(self, epoch):
        self.generator.manual_seed(42 + epoch)
        self.generator2.manual_seed(42 + epoch)

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
        sampler.set_epoch(epoch)
        air = torch.ones([214,16385], dtype=torch.complex64)
        mic = torch.ones([69,16385], dtype=torch.complex64)
        for x_orig, x_aug in tqdm.tqdm(loader):
            bat = x_orig.shape[0]
            with torch.no_grad():
                torch.set_num_threads(4)
                air_conv = air[torch.randint(0, 214, size=(bat,), dtype=torch.long)]
                mic_conv = mic[torch.randint(0, 69, size=(bat,), dtype=torch.long)]
                auga = torch.fft.rfft(x_aug, 16384*2, axis=1)
                b = torch.fft.irfft(auga * air_conv * mic_conv, 16384*2, axis=1)
                b = b[:,8000:16000]
                del air_conv
                del mic_conv
                del auga
