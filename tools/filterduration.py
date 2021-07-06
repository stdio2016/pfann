import argparse
import csv

argp = argparse.ArgumentParser()
argp.add_argument('--csv', required=True)
argp.add_argument('--min-len', type=float, default=0)
argp.add_argument('--max-len', type=float, default=1e999)
argp.add_argument('--out', required=True)
args = argp.parse_args()

out = []
with open(args.csv) as fin:
    reader = csv.reader(fin)
    out.append(next(reader))
    n = 0
    for row in reader:
        duration = float(row[1])
        n += 1
        if args.min_len <= duration <= args.max_len:
            out.append(row)
print('total %d sounds, filter remain %d sounds' % (n, len(out)-1))

with open(args.out, 'w', newline='\n') as fout:
    writer = csv.writer(fout)
    writer.writerows(out)