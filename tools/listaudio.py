import argparse
import os
import random
import miniaudio
import subprocess
import csv
import multiprocessing
import tqdm
from multiprocessing import Pool

argp = argparse.ArgumentParser()
argp.add_argument('--folder', required=True)
argp.add_argument('--sample', type=int)
argp.add_argument('--threads', type=int)
argp.add_argument('--out', default='out.csv')
args = argp.parse_args()

def ffmpeg_get_audio_length(filename):
    tmpname = 'tmp%d.wav' % os.getpid()
    if os.path.exists(tmpname):
        os.unlink(tmpname)
    subprocess.run(['ffmpeg', '-i', filename, '-y', tmpname],
        stderr=open(os.devnull),
        stdout=open(os.devnull),
        stdin=open(os.devnull))
    if os.path.exists(tmpname):
        info = miniaudio.get_file_info(tmpname)
        os.unlink(tmpname)
        return info.duration
    print('failed to decode %s. maybe the file is broken!' % filename)
    return None

def get_audio_length(filename):
    ext = os.path.splitext(filename)[1]
    if ext not in {'.wav', '.mp3', '.flac', '.ogg'}:
        #print('miniaudio cannot decode %s files. Try FFmpeg' % ext)
        return ffmpeg_get_audio_length(filename)
    try:
        if os.name == 'nt' and not filename.isascii():
            #print('%s contains non-ascii characters. Need to read by Python first' % filename)
            with open(filename, 'rb') as fin:
                code = fin.read()
            audio = miniaudio.decode(code)
        else:
            audio = miniaudio.read_file(filename, True)
        return audio.duration
    except miniaudio.DecodeError as x:
        #print('miniaudio cannot decode %s. Try FFmpeg' % filename)
        return ffmpeg_get_audio_length(filename)

formats = {'.wav', '.mp3', '.m4a', '.aac', '.ogg', '.flac', '.webm'}
def find_all_audio(folder, all_files):
    for name in os.listdir(folder):
        full_name = os.path.join(folder, name)
        if os.path.isdir(full_name):
            find_all_audio(full_name, all_files)
        else:
            ext = os.path.splitext(name)[1]
            if ext in formats:
                all_files.append(full_name)
    return all_files

def worker(filename):
    return filename, get_audio_length(filename)

if __name__ == '__main__':
    all_files = []
    print('searching audio files...')
    find_all_audio(args.folder, all_files)
    multiprocessing.set_start_method('spawn')
    with Pool(args.threads) as p:
        sound_files = []
        with tqdm.tqdm(total=len(all_files)) as pbar:
            for i, (filename, du) in enumerate(p.imap_unordered(worker, all_files)):
                if du is not None:
                    sound_files.append([filename, du])
                pbar.update()
    sound_files.sort()
    if args.sample:
        sound_files = random.sample(sound_files, args.sample)
    with open(args.out, 'w', encoding='utf8', newline='\n') as fout:
        writer = csv.writer(fout, lineterminator="\r\n")
        writer.writerow(['file', 'duration'])
        writer.writerows(sound_files)
