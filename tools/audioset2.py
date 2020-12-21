import argparse
import csv
import os
import subprocess
import time
from datetime import datetime

def yt_rename(name):
    new = []
    for ch in name:
        if ch.islower():
            new.append('=')
        new.append(ch)
    return ''.join(new)

def download(name, start, end, where, loger):
    out_name = '%s_%d.wav' % (name, start)
    out_path = os.path.join(where, out_name)
    if os.path.exists(out_path):
        return

    tmp_name = '%s_%d_tmp.wav' % (name, start)
    tmp_path = os.path.join(where, tmp_name)
    t1 = time.time()
    print('download %s from %d to %d' % (name, start, end))
    loger.write('%s download %s from %d to %d\n' % (datetime.now(), name, start, end))
    loger.flush()
    proc = subprocess.Popen(['youtube-dl', '-f', 'bestaudio',
        '--get-url', 'https://youtube.com/watch?v=%s' % name,
    ], stdout=subprocess.PIPE, stderr=loger)
    link = proc.stdout.read().strip()
    proc.wait()
    if proc.returncode == 0:
        proc = subprocess.Popen(['ffmpeg', '-loglevel', 'error',
            '-ss', str(start), '-i', link, '-t', str(end-start),
            '-y', out_path
        ], stdin=None, stderr=subprocess.PIPE)
        errs = proc.stderr.read().decode('utf8')
        print(errs, end='')
        loger.write(errs)
        if not os.path.exists(out_path):
            print('failed to download ;-(')
            loger.write('%s download %s error!\n' % (datetime.now(), name))
        loger.flush()
    else:
        print('failed to download ;-(')
        with open(out_path, 'wb') as fout:
            pass
    t2 = time.time()
    print('stop for a moment~~~')
    time.sleep(max(2, 10 - (t2-t1)))

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('csv')
    args.add_argument('folder')
    args = args.parse_args()

    os.makedirs(args.folder, exist_ok=True)

    with open(args.csv, 'r', encoding='utf8') as fin:
        reader = csv.reader(fin, skipinitialspace=True)
        segments = []
        nameuu = set()
        for item in reader:
            if item[0].startswith('#'): continue
            
            name = item[0]
            start = float(item[1])
            end = float(item[2])
            segments.append([name, start, end])
            nameuu.add(name.upper())
    print(len(nameuu), len(segments))

    loger = open('dlyt.txt', 'a')
    loger.write('%s start program...\n' % datetime.now())
    loger.flush()
    for name, start, end in segments:
        download(name, start, end, args.folder, loger)
    loger.write('%s end program...\n' % datetime.now())
    loger.close()
