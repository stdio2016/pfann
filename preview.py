import math
import warnings
import argparse

import torch
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import torchaudio

from datautil.dataset import build_data_loader
import simpleutils

if __name__ == '__main__':
    # don't delete this line, because my data loader uses queues
    torch.multiprocessing.set_start_method('spawn')
    args = argparse.ArgumentParser()
    args.add_argument('-d', '--data', required=True)
    args.add_argument('-p', '--params', default='configs/default.json')
    args = args.parse_args()
    
    params = simpleutils.read_config(args.params)
    
    train_data = build_data_loader(params, args.data)
    i = 0
    train_data.dataset.output_wav = True
    train_data.sampler.sampler.shuffle = False
    iterator = iter(train_data)
    for a in iterator:
        i += 1
        sound = a.transpose(0,1).flatten(1,2)
        sound *= 0.5 / torch.max(torch.abs(sound))
        torchaudio.save('temp%d.wav' % i, sound, 8000)
        print(i)
        if i >= 3:
            iterator._shutdown_workers()
            # kill my preloader
            train_data.sampler.sampler.preloader.terminate()
            break
    print('stopping...')
