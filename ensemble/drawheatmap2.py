import argparse
import csv
import math

import matplotlib
matplotlib.rcParams['font.family'] = ['Heiti TC']

import matplotlib.pyplot as plt
import seaborn

args = argparse.ArgumentParser()
args.add_argument('file')
args = args.parse_args()

with open(args.file) as fin:
    reader = csv.reader(fin)
    col_names = [x for x in next(reader)[1:]]
    row_names = []
    data = []
    for row in reader:
        row_names.append(float(row[0]))
        data.append([float(x) for x in row[1:]])

col_names = ['FMA\n-6dB', 'FMA\n-4dB', 'FMA\n-2dB', 'FMA\n0dB', 'FMA\n2dB', 'FMA\n4dB', 'FMA\n6dB', 'FMA\n8dB', 'FMA\n-10~10dB', 'MIREX']
row_names = ['$10^{%d}$' % math.log10(x) for x in row_names]

seaborn.set(font="Heiti TC", font_scale=0.5)

seaborn.heatmap(data, annot=True, xticklabels=col_names, yticklabels=row_names, fmt='.4f', cmap='viridis')
plt.xlabel('驗證資料集')
plt.ylabel('C')
plt.savefig(args.file + '.pdf')
plt.show()
