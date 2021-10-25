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

snrs = ['out2_snr-6', 'out2_snr-4', 'out2_snr-2', 'out2_snr0', 'out2_snr2', 'out2_snr4', 'out2_snr6', 'out2_snr8', 'out2', 'mirex']

dats = [['C']+snrs]
for C in ['0.01','0.1'] + [str(10**x) for x in range(0,11)]:
    dats.append([C])
    for snr in snrs:
        svm = 'lin_C' + C + '.pkl'
        a = np.load(args.nn_npy + snr + '.npy')
        b = np.load(args.lm_npy + snr + '.npy')
        select = a[:,1] + b[:,1] == 1
        x2 = np.stack([a[:,0], b[:,0]], axis=1)
        with open(os.path.join(args.svms, svm), 'rb') as fin:
            model = pickle.load(fin)
        pred = model.predict(x2)
        ok = np.where(pred, a[:,1], b[:,1])[select]
        acc = np.mean(ok)
        dats[-1].append(acc)
        #print('%s acc=%.4f' % (svm, acc))
with open(args.out, 'w', newline='\n') as fout:
    writer = csv.writer(fout)
    writer.writerows(dats)
