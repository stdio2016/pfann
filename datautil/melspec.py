import torch
import torchaudio

class MelSpec(torch.nn.Module):
    def __init__(self,
            sample_rate=8000,
            stft_n=1024,
            stft_hop=256,
            f_min=300,
            f_max=4000,
            n_mels=256,
            naf_mode=False,
            mel_log='log',
            spec_norm='l2'):
        super(MelSpec, self).__init__()
        self.naf_mode = naf_mode
        self.mel_log = mel_log
        self.spec_norm = spec_norm
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=stft_n,
            hop_length=stft_hop,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            window_fn=torch.hann_window,
            power = 1 if naf_mode else 2,
            pad_mode = 'constant' if naf_mode else 'reflect',
            norm = 'slaney' if naf_mode else None,
            mel_scale = 'slaney' if naf_mode else 'htk'
        )

    def forward(self, x):
        # normalize volume
        p = 1e999 if self.spec_norm == 'max' else 2
        x = torch.nn.functional.normalize(x, p=p, dim=-1)

        if self.naf_mode:
            x = self.mel(x) + 0.06
        else:
            x = self.mel(x) + 1e-8
        
        if self.mel_log == 'log10':
            x = torch.log10(x)
        elif self.mel_log == 'log':
            x = torch.log(x)
        
        if self.spec_norm == 'max':
            x = x - torch.amax(x, dim=(-2,-1), keepdim=True)
        return x

def build_mel_spec_layer(params):
    return MelSpec(
        sample_rate = params['sample_rate'],
        stft_n = params['stft_n'],
        stft_hop = params['stft_hop'],
        f_min = params['f_min'],
        f_max = params['f_max'],
        n_mels = params['n_mels'],
        naf_mode = params.get('naf_mode', False),
        mel_log = params.get('mel_log', 'log'),
        spec_norm = params.get('spec_norm', 'l2')
    )

if __name__ == '__main__':
    import simpleutils
    params = simpleutils.read_config('configs/default.json')
    mel = build_mel_spec_layer(params).cuda()
    x = torch.rand(2, 8000).cuda() - 0.5
    y = mel(x)
    print(y)
    print(y.shape)
