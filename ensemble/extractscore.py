import csv
import argparse
import os
import numpy as np

args = argparse.ArgumentParser()
args.add_argument('songlist')
args.add_argument('groundtruth')
args.add_argument('predict')
args.add_argument('out')
args = args.parse_args()

def extract_ans_txt(file):
    with open(file, 'r') as fin:
        out = []
        for line in fin:
            if line.endswith('\n'): line = line[:-1]
            query, ans = line.split('\t')
            my_query = os.path.splitext(os.path.split(query)[1])[0]
            my_ans = os.path.splitext(os.path.split(ans)[1])[0]
            out.append((my_query, my_ans))
    return out

def extract_ans_csv(file):
    with open(file, 'r') as fin:
        out = []
        reader = csv.reader(fin)
        next(reader)
        for line in reader:
            query, ans = line[:2]
            my_query = os.path.splitext(os.path.split(query)[1])[0]
            my_ans = os.path.splitext(os.path.split(ans)[1])[0]
            if my_query in out:
                print('Warning! query %s occured twice' % query)
            out.append((my_query, my_ans))
    return out

def extract_ans(file):
    if file.endswith('.csv'):
        return extract_ans_csv(file)
    return extract_ans_txt(file)

GT = dict(extract_ans(args.groundtruth))
PR = extract_ans(args.predict)

with open(args.songlist) as fin:
    song_list = []
    song_ids = {}
    for i, line in enumerate(fin):
        if line.endswith('\n'): line = line[:-1]
        line = os.path.splitext(os.path.split(line)[1])[0]
        song_list.append(line)
        song_ids[line] = i

sco_bin = np.fromfile(args.predict+'.bin', dtype=np.float32)
sco_bin = sco_bin.reshape([-1, len(song_list), 2])

scores = []
for i in range(len(PR)):
    query, ans = PR[i]
    if query in GT:
        real_ans = GT[query]
        sco = sco_bin[i, song_ids[ans], 0]
        scores.append((sco, ans==real_ans))
    else:
        print('query %s in prediction file not found!!' % query)
        print('ARE YOU KIDDING ME?')
        exit(1)
scores = np.array(scores, dtype=np.float32)
np.save(args.out, scores)
