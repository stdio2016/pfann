import argparse
import pickle

import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

args = argparse.ArgumentParser()
args.add_argument('lm_npy')
args.add_argument('nn_npy')
args.add_argument('--svm')
args.add_argument('--out')
args = args.parse_args()

a = np.load(args.nn_npy)
b = np.load(args.lm_npy)
select = a[:,1] + b[:,1] == 1
x = np.stack([a[select,0], b[select,0]], axis=1)
y = a[select,1]
x2 = np.stack([a[:,0], b[:,0]], axis=1)
print('nn wins', np.sum(y==1))
print('landmark wins', np.sum(y==0))

xx = np.linspace(0, 0.8, 200)
yy = np.linspace(0, 0.025, 200)
xx, yy = np.meshgrid(xx, yy)
if args.svm:
    with open(args.svm, 'rb') as fin:
        model = pickle.load(fin)
    pred = model.predict(x2)
    ok = np.where(pred, a[:,1], b[:,1])
    acc = np.mean(ok)
    print('acc=%.4f' % acc)

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    plt.contourf(xx, yy, Z.reshape(200, 200), np.arange(101)*0.01)
print('nn score too big', x[np.where(x[:,0] > 0.8)])
print('lm score too big', x[np.where(x[:,1] > 0.025)], y[np.where(x[:,1] > 0.025)])
plt.scatter(x[y==0, 0], x[y==0, 1])
plt.scatter(x[y==1, 0], x[y==1, 1])
plt.xlabel('neural network score')
plt.xlim(0, 0.8)
plt.ylabel('landmark score')
plt.ylim(0, 0.025)
if args.out:
    plt.savefig(args.out)
plt.show()
