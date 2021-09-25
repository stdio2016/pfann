import argparse
import pickle
import os

import numpy as np
from sklearn.svm import SVC

args = argparse.ArgumentParser()
args.add_argument('lm_npy')
args.add_argument('nn_npy')
args.add_argument('svms')
args = args.parse_args()

a = np.load(args.nn_npy)
b = np.load(args.lm_npy)
select = a[:,1] + b[:,1] == 1
x = np.stack([a[select,0], b[select,0]], axis=1)
y = a[select,1]
print('nn wins', np.sum(y==1))
print('landmark wins', np.sum(y==0))

x2 = np.stack([a[:,0], b[:,0]], axis=1)

for svm in sorted(os.listdir(args.svms)):
    if svm.endswith('.pkl'):
        with open(os.path.join(args.svms, svm), 'rb') as fin:
            model = pickle.load(fin)
        pred = model.predict(x2)
        ok = np.where(pred, a[:,1], b[:,1])
        acc = np.mean(ok)
        print('%s acc=%.4f' % (svm, acc))
