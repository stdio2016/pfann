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
from datautil.preprocess import preprocess_music
from simpleutils import read_config

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
    def __init__(self, params, train_val):
        # load some configs
        assert train_val in {'train', 'validate'}
        sample_rate = params['sample_rate']
        self.augmented = True
        self.segment_size = int(params['segment_size'] * sample_rate)
        self.hop_size = int(params['hop_size'] * sample_rate)
        self.time_offset = int(params['time_offset'] * sample_rate) # select 1.2s of audio, then choose two random 1s of audios
        self.pad_start = int(params['pad_start'] * sample_rate) # include more audio at the left of a segment to simulate reverb
        self.params = params
        
        # get fft size needed for reverb
        fftconv_n = 1024
        air_len = int(params['air']['length'] * sample_rate)
        ir_len = int(params['micirp']['length'] * sample_rate)
        fft_needed = self.segment_size + self.pad_start + air_len + ir_len
        while fftconv_n < fft_needed:
            fftconv_n *= 2
        self.fftconv_n = fftconv_n

        # datasets data augmentation
        cache_dir = params['cache_dir']
        os.makedirs(cache_dir, exist_ok=True)
        self.noise = NoiseData(noise_dir='/musdata/dataset/audioset', list_csv=params['noise'][train_val], sample_rate=sample_rate, cache_dir=cache_dir)
        self.air = AIR(air_dir='/musdata/dataset/AIR_1_4', list_csv=params['air'][train_val], length=params['air']['length'], fftconv_n=fftconv_n, sample_rate=sample_rate)
        self.micirp = MicIRP(mic_dir='/musdata/dataset/micirp', list_csv=params['micirp'][train_val], length=params['micirp']['length'], fftconv_n=fftconv_n, sample_rate=sample_rate)

        # Load music dataset as memory mapped file
        file_name = os.path.splitext(os.path.split(params[train_val + '_csv'])[1])[0]
        file_name = os.path.join(cache_dir, '1' + file_name)
        if os.path.exists(file_name + '.bin'):
            print('load cached music from %s.bin' % file_name)
        else:
            preprocess_music('/musdata/dataset/fma_medium', params[train_val + '_csv'], sample_rate, file_name)
        self.f = NumpyMemmapDataset(file_name + '.bin', np.int16)
        
        # some segmentation settings
        song_len = np.load(file_name + '.npy')
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
        shift_range = 0
        segment_size = self.pad_start + self.segment_size
        offset1 = torch.randint(high=shift_range+1, size=(len(x),)).tolist()
        offset2 = torch.randint(high=shift_range+1, size=(len(x),)).tolist()
        x_orig = [xi[off + self.pad_start : off + segment_size] for xi, off in zip(x, offset1)]
        x_orig = torch.Tensor(np.stack(x_orig).astype(np.float32))
        x_aug = [xi[off : off + segment_size] for xi, off in zip(x, offset2)]
        x_aug = torch.Tensor(np.stack(x_aug).astype(np.float32))
        
        # background noise
        x_aug = self.noise.add_noises(x_aug, self.params['noise']['snr_min'], self.params['noise']['snr_max'])
        
        # impulse response
        spec = torch.fft.rfft(x_aug, self.fftconv_n)
        if self.air is not None:
            spec *= self.air.random_choose(spec.shape[0])
        if self.micirp is not None:
            spec *= self.micirp.random_choose(spec.shape[0])
        x_aug = torch.fft.irfft(spec, self.fftconv_n)[..., self.pad_start:segment_size]
        
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
    def __init__(self, music_data: MusicSegmentDataset, shuffle_size):
        self.music_data = music_data
        self.shuffle_size = shuffle_size
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
    params = read_config('configs/default.json')
    model = FpNetwork(d=128, h=1024, u=32, F=256, T=32, params={'fuller':True})
    #model = model.cuda()
    dataset = MusicSegmentDataset(params, 'validate')
    shuffler = TwoStageShuffler(dataset, params['shuffle_size'])
    loader = DataLoader(dataset, sampler=BatchSampler(shuffler, 320, False), batch_size=None, num_workers=4, pin_memory=True, prefetch_factor=5)
    i = 0
    for epoch in range(2):
        for x in tqdm.tqdm(loader):
            i += 1
            if i == 1:
                wavs = x.permute(1, 0, 2).flatten(1, 2)
                wavs /= max(wavs.abs().max(), 1e-4)
                torchaudio.save('trylisten.wav', wavs[:,:8000*30], 8000)
            torch.set_num_threads(1)
            #x = x.cuda()
            if torch.any(x.isnan()): print('QQ')
            fake = torch.nn.functional.pad(x, (0, 192))
            fake = fake.reshape([-1, 256, 32])
            #model(fake)
