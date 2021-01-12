import json
import os
import subprocess
from pathlib import Path

import miniaudio
import numpy as np

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
        stdout=subprocess.PIPE)
    code = proc.stdout.read()
    if code:
        audio = miniaudio.decode(code, output_format=miniaudio.SampleFormat.FLOAT32)
        samples = np.array(audio.samples).reshape([-1, audio.nchannels]).T
        return samples, audio.sample_rate
    print('failed to decode %s. maybe the file is broken!' % filename)
    return np.array([0])

def get_audio(filename):
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
