import torch

class SpecAugment:
    def __init__(self, params):
        self.freq_min = params.get('cutout_min', 0.1) # 5
        self.freq_max = params.get('cutout_max', 0.5) # 20
        self.time_min = params.get('cutout_min', 0.1) # 5
        self.time_max = params.get('cutout_max', 0.5) # 16
        
        self.cutout_min = params.get('cutout_min', 0.1) # 0.1
        self.cutout_max = params.get('cutout_max', 0.5) # 0.4
    
    def get_mask(self, F, T):
        mask = torch.zeros(F, T)
        # cutout
        cutout_max = self.cutout_max
        cutout_min = self.cutout_min
        f = F * (cutout_min + torch.rand(1) * (cutout_max-cutout_min))
        f = int(f)
        f0 = torch.randint(0, F - f + 1, (1,))
        t = T * (cutout_min + torch.rand(1) * (cutout_max-cutout_min))
        t = int(t)
        t0 = torch.randint(0, T - t + 1, (1,))
        mask[f0:f0+f, t0:t0+t] = 1
        
        # frequency masking
        f = F * (self.freq_min + torch.rand(1) * (self.freq_max - self.freq_min))
        f = int(f)
        f0 = torch.randint(0, F - f + 1, (1,))
        mask[f0:f0+f, :] = 1
        
        # time masking
        t = T * (self.time_min + torch.rand(1) * (self.time_max - self.time_min))
        t = int(t)
        t0 = torch.randint(0, T - t + 1, (1,))
        mask[:, t0:t0+t] = 1
        return mask
    
    def augment(self, x):
        mask = self.get_mask(x.shape[-2], x.shape[-1])
        x = x * (1 - mask)
        return x
