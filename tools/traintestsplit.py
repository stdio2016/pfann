import argparse
import csv
import random

argp = argparse.ArgumentParser()
argp.add_argument('--csv', default='out.csv')
argp.add_argument('--train-size', type=int)
argp.add_argument('--train', default='train.csv')
argp.add_argument('--test-size', type=int)
argp.add_argument('--test', default='test.csv')
args = argp.parse_args()

with open(args.csv, 'r', encoding='utf8') as fin:
    reader = csv.reader(fin)
    data = []
    next(reader)
    for row in reader:
        data.append(row)

n = len(data)
if args.train_size is None:
    if args.test_size is None:
        args.train_size = n//2
    else:
        args.train_size = n - args.test_size
if args.test_size is None:
    args.test_size = n - args.train_size
print('There are %d data' % n)
assert args.train_size + args.test_size <= n, 'Not enough data for train/test split'

train_index = random.sample(list(range(n)), args.train_size)
less_index = list(set(range(n)) - set(train_index))
test_index = random.sample(less_index, args.test_size)
train_data = map(lambda x: data[x], train_index)

with open(args.train, 'w', encoding='utf8', newline='\n') as fout:
    writer = csv.writer(fout)
    writer.writerow(['file', 'duration'])
    writer.writerows(train_data)

test_data = map(lambda x: data[x], test_index)
with open(args.test, 'w', encoding='utf8', newline='\n') as fout:
    writer = csv.writer(fout)
    writer.writerow(['file', 'duration'])
    writer.writerows(test_data)
