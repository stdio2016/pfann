import json
import os
import subprocess
from pathlib import Path

import miniaudio
import numpy as np
import wave
import io

def miniaudio_get_audio(filename):
    code = Path(filename).read_bytes()
    audio = miniaudio.decode(code, output_format=miniaudio.SampleFormat.FLOAT32)
    samples = np.array(audio.samples).reshape([-1, audio.nchannels]).T
    return samples, audio.sample_rate

def ffmpeg_get_audio(filename):
    error_log = open(os.devnull, 'w')
    proc = subprocess.Popen(['ffmpeg', '-i', filename, '-f', 'wav', 'pipe:1'],
        stderr=error_log,
        stdin=open(os.devnull),
        stdout=subprocess.PIPE,
        bufsize=1000000)
    try:
        wav = wave.open(io.BytesIO(proc.stdout.read()))
        n = wav.getnframes()
        samples = np.frombuffer(wav.readframes(n), dtype=np.int16) / 32768
        samples = samples.reshape([-1, wav.getnchannels()]).T
        return samples, wav.getframerate()
    except (wave.Error, EOFError):
        print('failed to decode %s. maybe the file is broken!' % filename)
    return np.array([0])

def wave_get_audio(filename):
    with wave.open(filename, 'rb') as wav:
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
    try:
        return miniaudio_get_audio(filename)
    except Exception:
        pass
    return ffmpeg_get_audio(filename)

class MiniaudioStream:
    def __init__(self, fin, stream, sample_rate, nchannels):
        self.fin = fin
        self.stream = stream
        self.sample_rate = sample_rate
        self.nchannels = nchannels

if os.name == 'nt':
    import win32api
    def _get_filename_bytes(filename: str) -> bytes:
        filename2 = os.path.expanduser(filename)
        if not os.path.isfile(filename2):
            raise FileNotFoundError(filename)
        # short file name usually works, but I don't know...
        return win32api.GetShortPathName(filename2).encode('mbcs')
    miniaudio._get_filename_bytes = _get_filename_bytes

def miniaudio_stream_audio(filename):
    info = miniaudio.get_file_info(filename)
    sample_rate = info.sample_rate
    nchannels = info.nchannels
    stream = miniaudio.stream_file(filename, sample_rate=sample_rate, nchannels=nchannels)
    return MiniaudioStream(None, stream, sample_rate, nchannels)

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

def stream_audio(filename):
    try:
        return miniaudio_stream_audio(filename)
    except Exception:
        raise
    return ffmpeg_stream_audio(filename)
