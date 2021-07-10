import os
import time

import numpy as np
import torch
import torch.fft
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, Sampler, BatchSampler
import tqdm

from model import FpNetwork
from datautil.noise import NoiseData
from datautil.ir import AIR, MicIRP

class NumpyMemmapDataset(Dataset):
    def __init__(self, path, dtype):
        self.path = path
        self.f = np.memmap(self.path, dtype=dtype)
    
    def __len__(self):
        return len(self.f)
    
    def __getitem__(self, idx):
        return self.f[idx]
    
    # need these, otherwise the pickler will read the whole data in memory
    def __getstate__(self):
        return {'path': self.path, 'dtype': self.f.dtype}
    
    def __setstate__(self, d):
        self.__init__(d['path'], d['dtype'])

# data augmentation on music dataset
class MusicSegmentDataset(Dataset):
    def __init__(self, location: str, len_file: str):
        # Load music dataset as memory mapped file
        self.f = NumpyMemmapDataset(location, np.int16)
        
        self.augmented = True
        self.segment_size = 8000
        self.hop_size = 4000
        self.time_offset = int(8000 * 1.2)
        self.pad_start = 8000 # include more audio at the left of a segment to simulate reverb
        
        self.noise = NoiseData(noise_dir='/musdata/dataset/audioset', list_csv='lists/noise_train.csv', sample_rate=8000, cache_dir=None)
        self.air = AIR(air_dir='/musdata/dataset/AIR_1_4', list_csv='lists/air_train.csv', length=1, fftconv_n=32768, sample_rate=8000)
        self.micirp = MicIRP(mic_dir='/musdata/dataset/micirp', list_csv='lists/micirp_train.csv', length=0.5, fftconv_n=32768, sample_rate=8000)
        
        # some segmentation settings
        song_len = np.load(len_file)
        self.cues = [] # start location of segment i
        self.offset_left = [] # allowed left shift of segment i
        self.offset_right = [] # allowed right shift of segment i
        self.song_range = [] # range of song i is (start time, end time, start idx, end idx)
        t = 0
        for duration in song_len:
            num_segs = round((duration - self.segment_size + self.hop_size) / self.hop_size)
            start_cue = len(self.cues)
            for idx in range(num_segs):
                my_time = idx * self.hop_size
                self.cues.append(t + my_time)
                self.offset_left.append(my_time)
                self.offset_right.append(duration - my_time)
            end_cue = len(self.cues)
            self.song_range.append((t, t + duration, start_cue, end_cue))
            t += duration
        
        # convert to torch Tensor to allow sharing memory
        self.cues = torch.LongTensor(self.cues)
        self.offset_left = torch.LongTensor(self.offset_left)
        self.offset_right = torch.LongTensor(self.offset_right)
    
    def get_single_segment(self, idx):
        cue = int(self.cues[idx])
        left = int(self.offset_left[idx])
        right = int(self.offset_right[idx])
        
        # choose a segment
        # segment looks like:
        #            cue
        #             v
        # [ pad_start | segment_size |   ...  ]
        #             \---   time_offset   ---/
        segment = self.f[cue - min(left, self.pad_start): cue + min(right, self.time_offset)]
        segment = np.pad(segment, [max(0, self.pad_start-left), max(0, self.time_offset-right)])
        
        # convert 16 bit to float
        return segment * np.float32(1/32768)
    
    def __getitem__(self, indices):
        # collect all segments. It is faster to do batch processing
        x = [self.get_single_segment(i) for i in indices]
        
        # random time offset
        shift_range = self.time_offset - self.segment_size
        segment_size = self.pad_start + self.segment_size
        offset1 = torch.randint(high=shift_range, size=(len(x),)).tolist()
        offset2 = torch.randint(high=shift_range, size=(len(x),)).tolist()
        x_orig = [xi[off + self.pad_start : off + segment_size] for xi, off in zip(x, offset1)]
        x_orig = torch.Tensor(np.stack(x_orig).astype(np.float32))
        x_aug = [xi[off : off + segment_size] for xi, off in zip(x, offset2)]
        x_aug = torch.Tensor(np.stack(x_aug).astype(np.float32))
        
        # background noise
        noise = self.noise.random_choose(x_aug.shape[0], x_aug.shape[1])
        vol_x = (x_aug ** 2).mean(dim=1).sqrt()
        vol_noise = (noise ** 2).mean(dim=1).sqrt()
        snr = torch.rand(x_aug.shape[0]) * 10
        ratio = vol_x / vol_noise
        ratio = torch.where((vol_x == 0) | (vol_noise == 0), torch.tensor(1.0), ratio)
        ratio *= 10 ** -(snr / 20)
        x_aug = x_aug + ratio.unsqueeze(1) * noise
        
        # impulse response
        spec = torch.fft.rfft(x_aug, 32768)
        spec *= self.air.random_choose(spec.shape[0])
        spec *= self.micirp.random_choose(spec.shape[0])
        x_aug = torch.fft.irfft(spec, 32768)[..., self.pad_start:segment_size]
        
        # normalize volume
        x_orig = torch.nn.functional.normalize(x_orig, p=2, dim=1)
        x_aug = torch.nn.functional.normalize(x_aug, p=2, dim=1)
        
        # output [x1_orig, x1_aug, x2_orig, x2_aug, ...]
        return torch.stack([x_orig, x_aug], dim=1)
    
    def fan_si_le(self):
        raise NotImplementedError('煩死了')
    
    def zuo_bu_chu_lai(self):
        raise NotImplementedError('做不起來')
    
    def __len__(self):
        return len(self.cues)
    
    def get_num_songs(self):
        return len(self.song_range)
    
    def get_song_segments(self, song_id):
        return self.song_range[song_id][2:4]
    
    def preload_song(self, song_id):
        start, end, _, _ = self.song_range[song_id]
        return self.f[start : end].copy()

class TwoStageShuffler(Sampler):
    def __init__(self, music_data: MusicSegmentDataset):
        self.music_data = music_data
        self.shuffle_size = 5
        self.shuffle = True
        self.loaded = set()
    
    def __len__(self):
        return len(self.music_data)
    
    def preload(self, song_id):
        if song_id not in self.loaded:
            self.music_data.preload_song(song_id)
            self.loaded.add(song_id)
    
    def shuffling_iter(self):
        # shuffle song list first
        shuffle_song = torch.randperm(self.music_data.get_num_songs())
        
        # split song list into chunks
        chunks = torch.split(shuffle_song, self.shuffle_size)
        for nc, songs in enumerate(chunks):
            # sort songs to make preloader read more sequential
            songs = torch.sort(songs)[0].tolist()
            
            buf = []
            for song in songs:
                # collect segment ids
                seg_start, seg_end = self.music_data.get_song_segments(song)
                buf += list(range(seg_start, seg_end))
            
            if nc == 0:
                # load first chunk
                for song in songs:
                    self.preload(song)
            
            # shuffle segments from song chunk
            shuffle_segs = torch.randperm(len(buf))
            shuffle_segs = [buf[x] for x in shuffle_segs]
            preload_cnt = 0
            for i, idx in enumerate(shuffle_segs):
                # output shuffled segment idx
                yield idx
                #if nc == 0 and i < 30: print(idx)
                
                # preload next chunk
                while len(self.loaded) < len(shuffle_song) and preload_cnt * len(shuffle_segs) < (i+1) * self.shuffle_size:
                    song = shuffle_song[len(self.loaded)].item()
                    self.preload(song)
                    preload_cnt += 1
    
    def non_shuffling_iter(self):
        # just return 0 ... len(dataset)-1
        yield from range(len(self))
    
    def __iter__(self):
        if self.shuffle:
            return self.shuffling_iter()
        return self.non_shuffling_iter()

class SegmentedDataLoader:
    def __init__(self):
        pass

if __name__ == '__main__':
    import torchaudio
    torch.manual_seed(123)
    mp.set_start_method('spawn')
    #model = FpNetwork(d=128, h=1024, u=32, F=256, T=32, params={'fuller':True})
    #model = model.cuda()
    dataset = MusicSegmentDataset('cache2/fma_medium_train.bin', 'cache2/fma_medium_train_idx.npy')
    shuffler = TwoStageShuffler(dataset)
    loader = DataLoader(dataset, sampler=BatchSampler(shuffler, 320, False), batch_size=None, num_workers=4, pin_memory=False, prefetch_factor=5)
    i = 0
    for epoch in range(2):
        for x in tqdm.tqdm(loader):
            i += 1
            if i == 1:
                wavs = x.permute(1, 0, 2).flatten(1, 2)
                wavs = torch.nn.functional.normalize(wavs, p=1e999)
                torchaudio.save('trylisten.wav', wavs[:,:8000*30], 8000)
            torch.set_num_threads(1)
            #x = x.cuda()
            if torch.any(x.isnan()): print('QQ')
            fake = torch.nn.functional.pad(x, (0, 192))
            fake = fake.reshape([-1, 256, 32])
            #model(fake)
