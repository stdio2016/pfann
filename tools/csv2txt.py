import argparse
import csv
import os

args = argparse.ArgumentParser()
args.add_argument('csv')
args.add_argument('--dir', required=True)
args.add_argument('--out')
args = args.parse_args()

if not args.out:
    args.out = args.csv + '.txt'

with open(args.csv, 'r', encoding='utf8') as fin, open(args.out, 'w', encoding='utf8') as fout:
    reader = csv.reader(fin)
    next(reader)
    for row in reader:
        file_path = os.path.abspath(os.path.join(args.dir, row[0]))
        fout.write(file_path + '\n')
