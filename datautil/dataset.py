import argparse
import csv
from pathlib import Path
import warnings

import tqdm
import torch
import torch.fft
import torch.multiprocessing as mp
import torch.nn.functional as F
import numpy as np
# torchaudio currently (0.7) will throw warning that cannot be disabled
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import torchaudio
torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False

from simpleutils import get_hash, read_config
from datautil.audio import get_audio
from datautil.ir import AIR, MicIRP
from datautil.noise import NoiseData

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, train_csv, data_dir, noise_dir, air_dir, micirp_dir, params, for_train=True):
        hop_size=0.5
        clip_size=1.2
        sample_rate=8000
        clips_per_song=60
        sel_size=1
        pad_start=1
        super(MyDataset, self).__init__()
        self.clip_size = int(clip_size * sample_rate)
        self.sel_size = int(sel_size * sample_rate)
        self.hop_size = int(hop_size * sample_rate)
        self.pad_start = int(pad_start * sample_rate)
        self.sample_rate = sample_rate
        self.clips_per_song = clips_per_song
        self.augmented = True
        self.output_wav = False
        self.spec_augment = True
        self.data_dir = Path(data_dir)
        self.params = params
        train_val = 'train' if for_train else 'validate'
        if noise_dir:
            self.noise = NoiseData(noise_dir=noise_dir,
                list_csv=params['noise'][train_val],
                sample_rate=sample_rate, cache_dir=params['cache_dir'])
        else:
            self.noise = None
        if air_dir:
            self.air = AIR(air_dir=air_dir,
                list_csv=params['air'][train_val],
                length=params['air']['length'],
                fftconv_n=params['fftconv_n'], sample_rate=sample_rate)
        else:
            self.air = None
        if micirp_dir:
            self.micirp = MicIRP(mic_dir=micirp_dir,
                list_csv=params['micirp'][train_val],
                length=params['micirp']['length'],
                fftconv_n=params['fftconv_n'], sample_rate=sample_rate)
        else:
            self.micirp = None
        with open(train_csv, 'r', encoding='utf8') as fin:
            reader = csv.DictReader(fin)
            self.files = [f['file'] for f in reader]
        self.prepare_cache(params['cache_dir'], self.files)
    
    def prepare_cache(self, cache_dir, files):
        print('preprocessing music...')
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
        wave, smpRate = get_audio(self.data_dir/name)
        wave = torch.FloatTensor(wave)
        # stereo to mono
        wave = wave.mean(dim=0, keepdim=True)
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
        if type(index) == list:
            torch.set_num_threads(1)
            bat = len(index)
            mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=8000,
                n_fft=1024,
                hop_length=256,
                f_min=300,
                f_max=4000,
                n_mels=256,
                window_fn=torch.hann_window)
            wav1 = torch.zeros([bat, self.sel_size], dtype=torch.float32)
            if not self.augmented:
                for i,x in enumerate(index):
                    wav1[i] = self[x]
                with warnings.catch_warnings():
                    # torchaudio is still using deprecated function torch.rfft
                    warnings.simplefilter("ignore")
                    return torch.unsqueeze(torch.log(mel(wav1) + 1e-8), 1)
            
            wav2 = torch.zeros([bat, self.sel_size + self.pad_start], dtype=torch.float32)
            for i,x in enumerate(index):
                w1, w2 = self[x]
                wav1[i] = w1
                wav2[i] = w2
            
            # background mixing
            wav2 -= wav2.mean(dim=1).unsqueeze(1)
            amp = torch.sqrt((wav2**2).mean(dim=1))
            snr_max = self.params['noise']['snr_max']
            snr_min = self.params['noise']['snr_min']
            snr = snr_min + torch.rand(bat) * (snr_max - snr_min)
            if self.noise:
                noise = self.noise.random_choose(bat, wav2.shape[1])
                noise_amp = torch.sqrt((noise**2).mean(dim=1))
                wav2 += noise * (amp / noise_amp * torch.pow(10, -0.05*snr)).unsqueeze(1)
            else:
                wav2 = torch.normal(mean=wav2, std=(amp*torch.pow(10, -0.05*snr)).unsqueeze(1))
            
            # IR filters
            wav2_freq = torch.fft.rfft(wav2, self.params['fftconv_n'], dim=1)
            if self.air:
                wav2_freq *= self.air.random_choose(bat)
            if self.micirp:
                wav2_freq *= self.micirp.random_choose(bat)
            wav2 = torch.fft.irfft(wav2_freq, self.params['fftconv_n'], dim=1)
            wav2 = wav2[:,self.pad_start:self.pad_start+self.sel_size]
            
            # normalize volume
            wav1 -= wav1.mean(dim=1).unsqueeze(1)
            wav1 = F.normalize(wav1, p=2, dim=1)
            wav2 = F.normalize(wav2, p=2, dim=1)
            
            # Mel spectrogram
            if not self.output_wav:
                with warnings.catch_warnings():
                    # torchaudio is still using deprecated function torch.rfft
                    warnings.simplefilter("ignore")
                    wav1 = torch.log(mel(wav1) + 1e-8)
                    wav2 = torch.log(mel(wav2) + 1e-8)
                
                # SpecAugment
                if self.spec_augment:
                    cutout_min = self.params['cutout_min']
                    cutout_max = self.params['cutout_max']
                    
                    # cutout
                    f = wav1.shape[1] * (cutout_min + torch.rand(1) * (cutout_max-cutout_min))
                    f = int(f)
                    f0 = torch.randint(0, wav1.shape[1] - f, (1,))
                    t = wav1.shape[2] * (cutout_min + torch.rand(1) * (cutout_max-cutout_min))
                    t = int(t)
                    t0 = torch.randint(0, wav1.shape[2] - t, (1,))
                    wav1[:, f0:f0+f, t0:t0+t] = 0
                    wav2[:, f0:f0+f, t0:t0+t] = 0
                    
                    # frequency masking
                    f = wav1.shape[1] * (cutout_min + torch.rand(1) * (cutout_max-cutout_min))
                    f = int(f)
                    f0 = torch.randint(0, wav1.shape[1] - f, (1,))
                    wav1[:, f0:f0+f, :] = 0
                    wav2[:, f0:f0+f, :] = 0
                    
                    # time masking
                    t = wav1.shape[2] * (cutout_min + torch.rand(1) * (cutout_max-cutout_min))
                    t = int(t)
                    t0 = torch.randint(0, wav1.shape[2] - t, (1,))
                    wav1[:, :, t0:t0+t] = 0
                    wav2[:, :, t0:t0+t] = 0
            return torch.stack([wav1, wav2], dim=1)
        #print('I am %d and I have %d' % (os.getpid(), index))
        wave, pad_start, start, du = index
        wave = wave[start-pad_start:start+du].to(torch.float32)
        wave *= 1/32768
        if not self.augmented:
            return wave[pad_start:pad_start+self.sel_size]
        
        # time offset modulation
        pos = torch.randint(0, self.clip_size-self.sel_size, size=(2,))
        wav1 = wave[pad_start+pos[0] : pad_start+self.sel_size+pos[0]]
        wav2 = wave[max(0, pad_start+pos[1]-self.pad_start) : pad_start+self.sel_size+pos[1]]
        wav2 = F.pad(wav2, (self.pad_start+self.sel_size-len(wav2), 0))
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
        self.shuffle = True
    
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
        self.preloader = preloader
        if self.shuffle:
            shuffled = torch.randperm(n, generator=self.generator).tolist()
        else:
            shuffled = list(range(n))
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
            if self.shuffle:
                sub_shuffle = torch.randperm(len(segIds), generator=self.generator2).tolist()
            else:
                sub_shuffle = range(len(segIds))
            for i in sub_shuffle:
                # look up for segment position
                segId = int(segIds[i])
                partId = int(dataset.segToPart[segId])
                partPos = int(dataset.partPos[partId])
                segPos = int(dataset.segPos[segId])
                padInfo = posdict[partId]
                seg_start = padInfo['start'] + (segPos - partPos)
                seg_padStart = min(self.pad_start, padInfo['padStart'] + (segPos - partPos))
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

def collate_fn(x):
    return x[0]

def build_data_loader(params, data_dir, noise_dir, air_dir, micirp_dir, for_train=True):
    num_workers = params.get('num_workers', 2)
    list_csv = params['train_csv'] if for_train else params['validate_csv']
    
    dataset = MyDataset(train_csv=list_csv, data_dir=data_dir, params=params,
        noise_dir=noise_dir, air_dir=air_dir, micirp_dir=micirp_dir,
        for_train=for_train)
    sampler = MySampler(dataset, params['shuffle_size'])
    loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        sampler=torch.utils.data.sampler.BatchSampler(
            sampler,
            batch_size=params['batch_size']//2,
            drop_last=False),
        collate_fn=collate_fn
    )
    loader.mydataset = dataset
    loader.mysampler = sampler
    return loader

if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('-d', '--data', required=True)
    argp.add_argument('-p', '--params', default='configs/default.json')
    args = argp.parse_args()
    
    mp.set_start_method('spawn')
    params = read_config(args.params)
    loader = build_data_loader(params, args.data, None, None, None)
    print('test dataloader for training data')
    for epoch in range(3):
        loader.mysampler.set_epoch(epoch)
        for x in tqdm.tqdm(loader):
            pass
