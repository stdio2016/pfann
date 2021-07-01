import csv
import os
import warnings

import tqdm
import numpy as np
import torch
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import torchaudio

from simpleutils import get_hash
from datautil.audio import get_audio

class NoiseData:
    def __init__(self, noise_dir, list_csv, sample_rate, cache_dir):
        print('loading noise dataset')
        hashes = []
        with open(list_csv, 'r') as fin:
            reader = csv.reader(fin)
            noises = []
            firstrow = next(reader)
            for row in reader:
                noises.append(row[0])
                hashes.append(get_hash(row[0]))
        hash = get_hash(''.join(hashes))
        #self.data = self.load_from_cache(list_csv, cache_dir, hash)
        #if self.data is not None:
        #    print(self.data.shape)
        #    return
        data = []
        silence_threshold = 0
        self.names = []
        for name in tqdm.tqdm(noises):
            smp, smprate = get_audio(os.path.join(noise_dir, name))
            smp = torch.from_numpy(smp.astype(np.float32))
            
            # convert to mono
            smp = smp.mean(dim=0)
            
            # strip silence start/end
            abs_smp = torch.abs(smp)
            if torch.max(abs_smp) <= silence_threshold:
                print('%s too silent' % name)
                continue
            has_sound = (abs_smp > silence_threshold).to(torch.int)
            start = int(torch.argmax(has_sound))
            end = has_sound.shape[0] - int(torch.argmax(has_sound.flip(0)))
            smp = smp[max(start, 0) : end]
            
            resampled = torchaudio.transforms.Resample(smprate, sample_rate)(smp)
            data.append(resampled)
            self.names.append(name)
        self.data = torch.cat(data)
        self.boundary = [0] + [x.shape[0] for x in data]
        self.boundary = torch.LongTensor(self.boundary).cumsum(0)
        del data
        #self.save_to_cache(list_csv, cache_dir, hash, self.data)
        print(self.data.shape)
    
    def load_from_cache(self, list_csv, cache_dir, hash):
        loc = os.path.join(cache_dir, os.path.basename(list_csv) + '.npy')
        loc2 = os.path.join(cache_dir, os.path.basename(list_csv) + '.hash')
        if os.path.exists(loc) and os.path.exists(loc2):
            with open(loc2, 'r') as fin:
                read_hash = fin.read().strip()
            if read_hash != hash:
                return None
            print('cache hit!')
            return torch.from_numpy(np.fromfile(loc, dtype=np.float32))
        return None
    
    def save_to_cache(self, list_csv, cache_dir, hash, audio):
        os.makedirs(cache_dir, exist_ok=True)
        loc = os.path.join(cache_dir, os.path.basename(list_csv) + '.npy')
        loc2 = os.path.join(cache_dir, os.path.basename(list_csv) + '.hash')
        with open(loc2, 'w') as fout:
            fout.write(hash)
        print('save to cache')
        audio.numpy().tofile(loc)
    
    def random_choose(self, num, duration, out_name=False):
        indices = torch.randint(0, len(self.names), size=(num,), dtype=torch.long)
        out = torch.zeros([num, duration], dtype=torch.float32)
        for i in range(num):
            idx = indices[i].item()
            start = int(self.boundary[idx])
            end = int(self.boundary[idx+1])
            du = end - start
            if du >= duration:
                # select random segment
                start = start + torch.randint(0, du - duration + 1, size=(1,)).item()
                end = start + duration
                out[i] = self.data[start:end]
            else:
                # put entire noise file at center
                p_start = (duration - du)//2
                p_end = p_start + du
                out[i, p_start:p_end] = self.data[start:end]
        if out_name:
            return out, [self.names[x] for x in indices]
        return out
