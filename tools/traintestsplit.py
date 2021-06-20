import argparse
import csv
import random

argp = argparse.ArgumentParser()
argp.add_argument('--csv', default='out.csv')
argp.add_argument('--train-size', type=int)
argp.add_argument('--train', default='train.csv')
argp.add_argument('--test-size', type=int)
argp.add_argument('--test', default='test.csv')
argp.add_argument('-p', '--portion', action='store_true')
args = argp.parse_args()

random.seed(1)
with open(args.csv, 'r', encoding='utf8') as fin:
    reader = csv.reader(fin)
    data = []
    firstrow = next(reader)
    for row in reader:
        data.append(row)

n = len(data)
if args.portion:
    ab = args.train_size + args.test_size
    train_size = n * args.train_size // ab
    test_size = n - train_size
else:
    if args.train_size is None:
        if args.test_size is None:
            train_size = n//2
        else:
            train_size = n - args.test_size
    else:
        train_size = args.train_size
    if args.test_size is None:
        test_size = n - train_size
    else:
        test_size = args.test_size
print('There are %d data' % n)
assert train_size + test_size <= n, 'Not enough data for train/test split'

train_index = random.sample(list(range(n)), train_size)
train_index.sort()
less_index = list(set(range(n)) - set(train_index))
test_index = random.sample(less_index, test_size)
test_index.sort()
train_data = map(lambda x: data[x], train_index)

with open(args.train, 'w', encoding='utf8', newline='\n') as fout:
    writer = csv.writer(fout)
    if firstrow:
        writer.writerow(firstrow)
    writer.writerows(train_data)
print('train data: %d' % train_size)

test_data = map(lambda x: data[x], test_index)
with open(args.test, 'w', encoding='utf8', newline='\n') as fout:
    writer = csv.writer(fout)
    if firstrow:
        writer.writerow(firstrow)
    writer.writerows(test_data)
print('test data: %d' % test_size)
