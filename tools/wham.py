import argparse
import csv
import os
import random

import numpy as np
import julius
import miniaudio
import torch
import tqdm

TOTAL_HOURS = 2.3
TOTAL_SECS = TOTAL_HOURS * 3600
NEW_SAMPLE_RATE = 8000

def gen_clips(noise_dir, noises, out_dir, out_type, total_secs):
    longs = 0
    wham_list = []
    out_dir = os.path.join(out_dir, out_type)
    os.makedirs(out_dir, exist_ok=True)
    with tqdm.tqdm(total=total_secs) as t:
        for name in noises:
            info = miniaudio.wav_read_file_f32(os.path.join(noise_dir, name))
            du = info.duration
            wham_list.append([os.path.join(out_type, name), du])
            longs += du
            with open(os.path.join(noise_dir, name), 'rb') as fin:
                code = fin.read()
            with open(os.path.join(out_dir, name), 'wb') as fout:
                fout.write(code)
            if longs >= total_secs:
                break
            t.update(du)
    with open(os.path.join(out_dir, 'list.csv'), 'w', encoding='utf8', newline='\n') as fout:
        writer = csv.writer(fout)
        writer.writerows(wham_list)
    return wham_list

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--wham', required=True)
    args.add_argument('--audioset', required=True)
    args = args.parse_args()
    
    wham_dir = os.path.join(args.wham, 'tr')
    noises = os.listdir(wham_dir)
    random.shuffle(noises)
    lst = gen_clips(wham_dir, noises, args.audioset, 'tr', TOTAL_SECS * 0.8)
    
    wham_dir = os.path.join(args.wham, 'cv')
    noises = os.listdir(wham_dir)
    random.shuffle(noises)
    lst = gen_clips(wham_dir, noises, args.audioset, 'cv', TOTAL_SECS * 0.2)
