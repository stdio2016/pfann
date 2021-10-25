import argparse
import re
from collections import Counter

args = argparse.ArgumentParser()
args.add_argument('log')
args = args.parse_args()

total_times = Counter()
with open(args.log, encoding='utf8') as fin:
    for line in fin:
        split = line.rfind('] ')
        if split == -1:
            body = line
        else:
            body = line[split+2:]
        for task in ['load', 'resample', 'stereo to mono', 'compute embedding', 'search', 'rerank', 'output answer', 'total query time']:
            s = re.search(task + r' (\d+\.\d+)s', body)
            if s:
                secs = float(s[1])
                total_times[task] += secs
for task in total_times:
    print('%s %.3f s' % (task, total_times[task]))
