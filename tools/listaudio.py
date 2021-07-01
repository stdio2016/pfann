import argparse
import csv
import multiprocessing
from multiprocessing import Pool
import os
import random
import subprocess
import wave

import tqdm

argp = argparse.ArgumentParser()
argp.add_argument('--folder', required=True)
argp.add_argument('--sample', type=int)
argp.add_argument('--threads', type=int)
argp.add_argument('--out', default='out.csv')
args = argp.parse_args()

class HackExtensibleWave:
    def __init__(self, stream):
        self.stream = stream
        self.pos = 0
    def read(self, n):
        r = self.stream.read(n)
        new_pos = self.pos + len(r)
        if self.pos < 20 and self.pos + n >= 20:
            r = r[:20-self.pos] + b'\x01\x00'[:new_pos-20] + r[22-self.pos:]
        elif 20 <= self.pos < 22:
            r = b'\x01\x00'[self.pos-20:new_pos-20] + r[22-self.pos:]
        self.pos = new_pos
        return r

def ffmpeg_get_audio_length(filename):
    proc = subprocess.Popen(['ffmpeg', '-i', filename, '-f', 'wav', 'pipe:1'],
        stderr=open(os.devnull),
        stdin=open(os.devnull),
        stdout=subprocess.PIPE,
        bufsize=1000000)
    try:
        wav = wave.open(HackExtensibleWave(proc.stdout))
        smprate = wav.getframerate()
        has = 1
        n = 0
        while has:
            has = len(wav.readframes(1000000))
            n += has
        n //= wav.getsampwidth() * wav.getnchannels()
        #smprate, wav = scipy.io.wavfile.read(proc.stdout)
        return n / smprate, smprate, wav.getnchannels()
        return wav.shape[0] / smprate, smprate, wav.shape[1]
    except (wave.Error, EOFError) as x:
        try:
            n = os.stat(filename).st_size
            if n == 0:
                print('file %s is empty!' % filename)
            else:
                print('failed to decode %s. maybe the file is broken!' % filename)
        except:
            print('failed to stat %s. maybe it is not a file anymore!' % filename)
    return None

def get_audio_length(filename):
    ext = os.path.splitext(filename)[1]
    return ffmpeg_get_audio_length(filename)

formats = {'.wav', '.mp3', '.m4a', '.aac', '.ogg', '.flac', '.webm'}
def find_all_audio(folder, relative, all_files):
    for name in os.listdir(folder):
        full_name = os.path.join(folder, name)
        nxt_relative = os.path.join(relative, name)
        if os.path.isdir(full_name):
            find_all_audio(full_name, nxt_relative, all_files)
        else:
            ext = os.path.splitext(name)[1]
            if ext in formats:
                all_files.append(nxt_relative)
    return all_files

def worker(filename):
    folder, relative = filename
    return relative, get_audio_length(os.path.join(folder, relative))

if __name__ == '__main__':
    all_files = []
    print('searching audio files...')
    find_all_audio(args.folder, '', all_files)
    all_files = [(args.folder, x) for x in all_files]
    multiprocessing.set_start_method('spawn')
    with Pool(args.threads) as p:
        sound_files = []
        with tqdm.tqdm(total=len(all_files)) as pbar:
            for i, (filename, du) in enumerate(p.imap_unordered(worker, all_files)):
                if du is not None:
                    sound_files.append([filename, *du])
                pbar.update()
    sound_files.sort()
    if args.sample:
        sound_files = random.sample(sound_files, args.sample)
    with open(args.out, 'w', encoding='utf8', newline='\n') as fout:
        if args.out.endswith('.csv'):
            # csv format with duration info
            writer = csv.writer(fout, lineterminator="\r\n")
            writer.writerow(['file', 'duration', 'sample_rate', 'channels'])
            writer.writerows(sound_files)
        else:
            # plain text list
            for sound_name, duration, smprate, nchannels in sound_files:
                fout.write(sound_name + '\n')
