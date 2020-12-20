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
    proc = subprocess.Popen(['ffmpeg', '-i', filename, '-f', 'wav', 'pipe:1'],
        stderr=subprocess.PIPE,
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
