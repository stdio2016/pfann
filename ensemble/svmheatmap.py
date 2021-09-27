import argparse
import csv
import os
import pickle

import numpy as np
from sklearn.svm import SVC

args = argparse.ArgumentParser()
args.add_argument('lm_npy')
args.add_argument('nn_npy')
args.add_argument('svms')
args.add_argument('out')
args = args.parse_args()

a = np.load(args.nn_npy)
b = np.load(args.lm_npy)
select = a[:,1] + b[:,1] == 1
x = np.stack([a[select,0], b[select,0]], axis=1)
y = a[select,1]
#print('nn wins', np.sum(y==1))
#print('landmark wins', np.sum(y==0))

x2 = np.stack([a[:,0], b[:,0]], axis=1)

gammas = ['1e-09','1e-08','1e-07','1e-06','1e-05','0.0001','0.001','0.01','0.1','1','10','100','1000']

dats = [['']+gammas]
for C in ['0.01','0.1'] + [str(10**x) for x in range(0,11)]:
    dats.append([C])
    for gamma in gammas:
        svm = 'rbf_C' + C + '_gamma' + gamma + '.pkl'
        with open(os.path.join(args.svms, svm), 'rb') as fin:
            model = pickle.load(fin)
        pred = model.predict(x2)
        ok = np.where(pred, a[:,1], b[:,1])
        acc = np.mean(ok)
        dats[-1].append(acc)
        #print('%s acc=%.4f' % (svm, acc))
with open(args.out, 'w', newline='\n') as fout:
    writer = csv.writer(fout)
    writer.writerows(dats)
