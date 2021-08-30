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
            out[my_query] = my_ans, 0
    return out

def extract_ans_csv(file):
    with open(file, 'r') as fin:
        out = {}
        reader = csv.reader(fin)
        next(reader)
        for line in reader:
            query, ans = line[:2]
            my_query = os.path.splitext(os.path.split(query)[1])[0]
            my_ans = os.path.splitext(os.path.split(ans)[1])[0]
            if my_query in out:
                print('Warning! query %s occured twice' % query)
            out[my_query] = my_ans, float(line[2])
    return out

def extract_ans(file):
    if file.endswith('.csv'):
        return extract_ans_csv(file)
    return extract_ans_txt(file)

GT = extract_ans(args.groundtruth)
PR = extract_ans(args.predict)

correct = 0
total = 0
scores = []
for query in PR:
    ans, sco = PR[query]
    if query in GT:
        real_ans, _ = GT[query]
        total += 1
        if ans == real_ans:
            correct += 1
        scores.append((sco, ans==real_ans))
    else:
        print('query %s in prediction file not found!!' % query)
        print('ARE YOU KIDDING ME?')
        exit(1)
print('song correct %d acc %.2f' % (correct, correct/total * 100))
scores.sort()
if correct == 0:
    print('totally wrong')
elif correct == total:
    print('all correct')
else:
    thres = (scores[total-correct-1][0] + scores[total-correct][0]) / 2
    FN = 0
    for sco, ok in scores:
        if sco > thres: break
        FN += ok
    print('threshold %f TP %d FN %d' % (thres, correct - FN, FN))
