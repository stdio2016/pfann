import argparse
import csv
import os

args = argparse.ArgumentParser()
args.add_argument('groundtruth')
args.add_argument('predict')
args = args.parse_args()

with open(args.groundtruth, 'r') as fin:
    reader = csv.DictReader(fin)
    gt = {}
    for row in reader:
        name = os.path.basename(row['query'])
        gt[name] = row

with open(args.predict, 'r') as fin:
    reader = csv.DictReader(fin)
    predict = list(reader)
    total = 0
    correct = 0
    correct_near = 0
    correct_exact = 0
    fail_time = []
    all_time = []
    for row in predict:
        name = os.path.basename(row['query'])
        ans = os.path.basename(row['answer'])
        actual = gt[name]
        actual_ans = os.path.basename(actual['answer'])
        total += 1
        tm = float(row['time'])
        actual_tm = float(actual['time'])
        if actual_ans == ans:
            correct += 1
            if abs(actual_tm - tm) <= 0.25:
                correct_exact += 1
            if abs(actual_tm - tm) <= 0.5:
                correct_near += 1
        else:
            fail_time.append(actual_tm % 0.5)
        all_time.append(actual_tm % 0.5)
print("exact match correct %d acc %f" % (correct_exact, correct_exact/total))
print("near match correct %d acc %f" % (correct_near, correct_near/total))
print("song correct %d acc %f" % (correct, correct/total))
