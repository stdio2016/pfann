import torch
from torch.utils.data import DataLoader, BatchSampler
from datautil.dataset_v2 import TwoStageShuffler

def make_false_data(N, F_bin, T):
    mock = torch.rand([N, F_bin, T], dtype=torch.float32)
    mock2 = mock + torch.rand([N, F_bin, T], dtype=torch.float32) * 1 - 0.5
    mock = torch.stack([mock, mock2], dim=1)
    return mock

# use this when you don't have training data
class MockedDataLoader:
    def __init__(self, train_val, configs, num_workers=4, pin_memory=False, prefetch_factor=2):
        assert train_val in {'train', 'validate'}
        F_bin = configs['n_mels']
        segn = int(configs['segment_size'] * configs['sample_rate'])
        T = (segn + configs['stft_hop'] - 1) // configs['stft_hop']
        # 1/50 of real training data
        num_fake_data = 0
        if train_val == 'train':
            num_fake_data = 584183//50
        else:
            num_fake_data = 29215//50
        self.dataset = make_false_data(num_fake_data, F_bin, T)
        assert configs['batch_size'] % 2 == 0
        self.batch_size = configs['batch_size']
        self.shuffler = TwoStageShuffler(self.dataset, configs['shuffle_size'])
        self.sampler = BatchSampler(self.shuffler, self.batch_size//2, False)
        self.num_workers = num_workers
        self.configs = configs
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

        # you can change shuffle to True/False
        self.shuffle = True
        # you can change augmented to True/False
        self.augmented = True
        # you can change eval time shift to True/False
        self.eval_time_shift = False

        self.loader = DataLoader(
            self.dataset,
            sampler=self.sampler,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor
        )

    def set_epoch(self, epoch):
        self.shuffler.set_epoch(epoch)

    def __iter__(self):
        #self.dataset.augmented = self.augmented
        #self.dataset.eval_time_shift = self.eval_time_shift
        self.shuffler.shuffle = self.shuffle
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)
