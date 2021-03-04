import csv
import random

dummys = set()
querys = []
with open('configs/fma_full.csv', 'r') as fin:
    reader = csv.reader(fin)
    next(reader)
    for row in reader:
        du = float(row[1])
        if du > 3600 or du < 30:
            continue
        dummys.add(row[0])

with open('configs/test.csv', 'r') as fin:
    reader = csv.reader(fin)
    next(reader)
    for row in reader:
        du = float(row[1])
        dummys.discard(row[0])
        querys.append(row[0])

dummys = list(dummys)
random.shuffle(dummys)
dummys = dummys[0:100000]
dummys.sort()
querys.sort()

with open('build_full_list.txt', 'w') as fout:
    for x in dummys:
        fout.write('/musdata/dataset/fma_full/' + x + '\n')
    for x in querys:
        fout.write('/musdata/dataset/fma_medium/' + x + '\n')
