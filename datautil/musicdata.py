import julius
import numpy as np
import torch
import torch.nn.functional as F

from datautil.audio import stream_audio

class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, params):
        self.params = params
        self.sample_rate = self.params['sample_rate']
        self.segment_size = int(self.sample_rate * self.params['segment_size'])
        self.hop_size = int(self.sample_rate * self.params['hop_size'])
        self.frame_shift_mul = self.params['indexer'].get('frame_shift_mul', 1)
        with open(file_list, 'r', encoding='utf8') as fin:
            self.files = []
            for x in fin:
                if x.endswith('\n'):
                    x = x[:-1]
                self.files.append(x)
    
    def __getitem__(self, index):
        smprate = self.sample_rate
        
        # resample
        stm = stream_audio(self.files[index])
        resampler = julius.ResampleFrac(stm.sample_rate, smprate)
        arr = []
        n = 0
        total = 0
        minute = stm.sample_rate * 60
        second = stm.sample_rate
        new_min = smprate * 60
        new_sec = smprate
        strip_head = 0
        wav = []
        for b in stm.stream:
            b = np.array(b).reshape([-1, stm.nchannels])
            b = np.mean(b, axis=1, dtype=np.float32) * (1/32768)
            arr.append(b)
            n += b.shape[0]
            total += b.shape[0]
            if n >= minute:
                arr = np.concatenate(arr)
                b = arr[:minute]
                out = torch.from_numpy(b)
                wav.append(resampler(out)[strip_head : new_min-new_sec//2])
                arr = [arr[minute-second:].copy()]
                strip_head = new_sec//2
                n -= minute-second
        # resample tail part
        arr = np.concatenate(arr)
        out = torch.from_numpy(arr)
        wav.append(resampler(out)[strip_head : ])
        wav = torch.cat(wav)

        if wav.shape[0] < self.segment_size:
            # this "music" is too short and need to be extended
            wav = F.pad(wav, (0, self.segment_size - wav.shape[0]))
        
        # normalize volume
        wav = wav.unfold(0, self.segment_size, self.hop_size//self.frame_shift_mul)
        wav = wav - wav.mean(dim=1).unsqueeze(1)
        wav = F.normalize(wav, p=2, dim=1)
        
        return index, self.files[index], wav
    
    def __len__(self):
        return len(self.files)
