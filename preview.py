import warnings
import argparse
import torch
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import torchaudio
import math
from datautil.dataset import build_data_loader

if __name__ == '__main__':
    # don't delete this line, because my data loader uses queues
    torch.multiprocessing.set_start_method('spawn')
    args = argparse.ArgumentParser()
    args.add_argument('--csv', required=True)
    args.add_argument('--cache-dir', required=True)
    args = args.parse_args()
    train_data = build_data_loader(
        csv_path=args.csv,
        cache_dir=args.cache_dir,
        num_workers=2, chunk_size=20000, batch_size=640)
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
