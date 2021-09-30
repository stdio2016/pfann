import argparse
import csv
import math

import matplotlib.pyplot as plt
import seaborn

args = argparse.ArgumentParser()
args.add_argument('file')
args = args.parse_args()

with open(args.file) as fin:
    reader = csv.reader(fin)
    col_names = [float(x) for x in next(reader)[1:]]
    row_names = []
    data = []
    for row in reader:
        row_names.append(float(row[0]))
        data.append([float(x) for x in row[1:]])

col_names = ['$10^{%d}$' % math.log10(x) for x in col_names]
row_names = ['$10^{%d}$' % math.log10(x) for x in row_names]

seaborn.set(font_scale=0.5)

seaborn.heatmap(data, annot=True, xticklabels=col_names, yticklabels=row_names, fmt='.4f', cmap='viridis')
plt.xlabel('gamma')
plt.ylabel('C')
plt.savefig(args.file + '.pdf')
plt.show()
