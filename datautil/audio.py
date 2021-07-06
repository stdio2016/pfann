import json
import os
import subprocess
from pathlib import Path

import numpy as np
import wave
import io

# because builtin wave won't read wav files with more than 2 channels
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

def ffmpeg_get_audio(filename):
    error_log = open(os.devnull, 'w')
    proc = subprocess.Popen(['ffmpeg', '-i', filename, '-f', 'wav', 'pipe:1'],
        stderr=error_log,
        stdin=open(os.devnull),
        stdout=subprocess.PIPE,
        bufsize=1000000)
    try:
        dat = proc.stdout.read()
        wav = wave.open(HackExtensibleWave(io.BytesIO(dat)))
        ch = wav.getnchannels()
        rate = wav.getframerate()
        n = wav.getnframes()
        dat = wav.readframes(n)
        del wav
        samples = np.frombuffer(dat, dtype=np.int16) / 32768
        samples = samples.reshape([-1, ch]).T
        return samples, rate
    except (wave.Error, EOFError):
        print('failed to decode %s. maybe the file is broken!' % filename)
    return np.zeros([1, 0]), 44100

def wave_get_audio(filename):
    with open(filename, 'rb') as fin:
        wav = wave.open(HackExtensibleWave(fin))
        smpwidth = wav.getsampwidth()
        if smpwidth not in {1, 2, 3}:
            return None
        n = wav.getnframes()
        if smpwidth == 1:
            samples = np.frombuffer(wav.readframes(n), dtype=np.uint8) / 128 - 1
        elif smpwidth == 2:
            samples = np.frombuffer(wav.readframes(n), dtype=np.int16) / 32768
        elif smpwidth == 3:
            a = np.frombuffer(wav.readframes(n), dtype=np.uint8)
            samples = np.stack([a[0::3], a[1::3], a[2::3], -(a[2::3]>>7)], axis=1).view(np.int32).squeeze(1)
            del a
            samples = samples / 8388608
        samples = samples.reshape([-1, wav.getnchannels()]).T
        return samples, wav.getframerate()

def get_audio(filename):
    if filename.endswith('.wav'):
        try:
            a = wave_get_audio(filename)
            if a: return a
        except Exception:
            pass
    return ffmpeg_get_audio(filename)

class FfmpegStream:
    def __init__(self, proc, sample_rate, nchannels):
        self.proc = proc
        self.sample_rate = sample_rate
        self.nchannels = nchannels
        self.stream = self.gen_stream()
    def __del__(self):
        self.proc.terminate()
        self.proc.communicate()
        del self.proc
    def gen_stream(self):
        num = yield np.array([], dtype=np.int16)
        if not num: num = 1024
        while True:
            to_read = num * self.nchannels * 2
            dat = self.proc.stdout.read(to_read)
            num = yield np.frombuffer(dat, dtype=np.int16)
            if not num: num = 1024
            if len(dat) < to_read:
                break

def ffmpeg_stream_audio(filename):
    proc = subprocess.Popen(['ffprobe', '-i', filename, '-show_streams',
            '-select_streams', 'a', '-print_format', 'json'],
        stderr=open(os.devnull, 'w'),
        stdin=open(os.devnull),
        stdout=subprocess.PIPE)
    prop = json.loads(proc.stdout.read())
    sample_rate = int(prop['streams'][0]['sample_rate'])
    nchannels = prop['streams'][0]['channels']
    proc = subprocess.Popen(['ffmpeg', '-i', filename,
            '-f', 's16le', '-acodec', 'pcm_s16le', 'pipe:1'],
        stderr=open(os.devnull, 'w'),
        stdin=open(os.devnull),
        stdout=subprocess.PIPE)
    return FfmpegStream(proc, sample_rate, nchannels)

class WaveStream:
    def __init__(self, filename):
        self.file = open(filename, 'rb')
        self.wave = wave.open(HackExtensibleWave(self.file))
        self.smpsize = self.wave.getnchannels() * self.wave.getsampwidth()
        self.sample_rate = self.wave.getframerate()
        self.nchannels = self.wave.getnchannels()
        if self.wave.getsampwidth() != 2:
            raise NotImplementedError('wave stream currently only supports 16bit wav')
        self.stream = self.gen_stream()
    def gen_stream(self):
        num = yield np.array([], dtype=np.int16)
        if not num: num = 1024
        while True:
            dat = self.wave.readframes(num)
            num = yield np.frombuffer(dat, dtype=np.int16)
            if not num: num = 1024
            if len(dat) < num * self.smpsize:
                break

def stream_audio(filename):
    try:
        return WaveStream(filename)
    except:
        pass
    return ffmpeg_stream_audio(filename)
