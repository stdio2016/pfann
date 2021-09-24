import argparse
import pickle
import os

import numpy as np
from sklearn.svm import SVC

args = argparse.ArgumentParser()
args.add_argument('lm_npy')
args.add_argument('nn_npy')
args.add_argument('out')
args = args.parse_args()

a = np.load(args.nn_npy)
b = np.load(args.lm_npy)
select = a[:,1] + b[:,1] == 1
x = np.stack([a[select,0], b[select,0]], axis=1)
y = a[select,1]
print('nn wins', np.sum(y==1))
print('landmark wins', np.sum(y==0))

print('Linear SVM')
for C in range(-2, 11):
    model = SVC(C=10**C, kernel='linear')
    model.fit(x, y)
    acc = np.mean(model.predict(x) == y)
    print('C={} train acc={:.4f}'.format(10**C, acc))
    with open(os.path.join(args.out, 'lin_C{}.pkl'.format(10**C)), 'wb') as fout:
        pickle.dump(model, fout)

print('RBF SVM')
for C in range(-2, 11):
    for gamma in range(-9, 4):
        model = SVC(C=10**C, kernel='rbf', gamma=10**gamma)
        model.fit(x, y)
        acc = np.mean(model.predict(x) == y)
        print('C={} gamma={} train acc={:.4f}'.format(10**C, 10**gamma, acc))
        with open(os.path.join(args.out, 'rbf_C{}_gamma{}.pkl'.format(10**C, 10**gamma)), 'wb') as fout:
            pickle.dump(model, fout)
