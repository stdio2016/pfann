import csv
import random

dummys = set()
with open('lists/fma_large.csv', 'r') as fin:
    reader = csv.reader(fin)
    next(reader)
    for row in reader:
        du = float(row[1])
        if du < 29.9:
            continue
        dummys.add(row[0])

with open('lists/fma_medium_train.csv', 'r') as fin:
    reader = csv.reader(fin)
    next(reader)
    for row in reader:
        du = float(row[1])
        dummys.discard(row[0])

vals = []
with open('lists/fma_medium_val.csv', 'r') as fin:
    reader = csv.reader(fin)
    next(reader)
    for row in reader:
        dummys.discard(row[0])
        vals.append(row[0])

tests = []
with open('lists/fma_medium_test.csv', 'r') as fin:
    reader = csv.reader(fin)
    next(reader)
    for row in reader:
        dummys.discard(row[0])
        tests.append(row[0])

dummys = list(dummys)
random.seed(3)
random.shuffle(dummys)
dummys = dummys[0:10000]
dummys.sort()
vals.sort()
tests.sort()

with open('lists/fma_out1.txt', 'w') as fout:
    for x in dummys:
        fout.write('../pfann_dataset/fma_large/' + x + '\n')
    for x in vals:
        fout.write('../pfann_dataset/fma_medium/' + x + '\n')

with open('lists/fma_out2.txt', 'w') as fout:
    for x in dummys:
        fout.write('../pfann_dataset/fma_large/' + x + '\n')
    for x in tests:
        fout.write('../pfann_dataset/fma_medium/' + x + '\n')
