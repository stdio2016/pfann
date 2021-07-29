import csv
import argparse
import os

args = argparse.ArgumentParser()
args.add_argument('groundtruth')
args.add_argument('predict')
args = args.parse_args()

def extract_ans_txt(file):
    with open(file, 'r') as fin:
        out = {}
        for line in fin:
            if line.endswith('\n'): line = line[:-1]
            query, ans = line.split('\t')
            my_query = os.path.splitext(os.path.split(query)[1])[0]
            my_ans = os.path.splitext(os.path.split(ans)[1])[0]
            if my_query in out:
                print('Warning! query %s occured twice' % query)
            out[my_query] = my_ans
    return out

def extract_ans_csv(file):
    with open(file, 'r') as fin:
        out = {}
        reader = csv.reader(fin)
        for line in reader:
            query, ans = line[:2]
            my_query = os.path.splitext(os.path.split(query)[1])[0]
            my_ans = os.path.splitext(os.path.split(ans)[1])[0]
            if my_query in out:
                print('Warning! query %s occured twice' % query)
            out[my_query] = my_ans
    return out

def extract_ans(file):
    if file.endswith('.csv'):
        return extract_ans_csv(file)
    return extract_ans_txt(file)

GT = extract_ans(args.groundtruth)
PR = extract_ans(args.predict)

correct = 0
total = 0
for query in PR:
    ans = PR[query]
    if query in GT:
        real_ans = GT[query]
        total += 1
        if ans == real_ans:
            correct += 1
    else:
        print('query %s in prediction file not found!!' % query)
        print('ARE YOU KIDDING ME?')
        exit(1)
print('song correct %d acc %f' % (correct, correct/total))
