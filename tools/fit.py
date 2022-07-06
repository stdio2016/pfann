from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

lm_score = [71.10, 79.65, 86.85, 91.10, 93.30, 95.20, 96.60, 97.70]
nn_score = [59.05, 75.20, 86.40, 92.55, 95.95, 97.30, 98.05, 99.00]
nn_score2 = [81.70, 89.55, 93.30, 95.60, 97.30, 98.10, 98.60, 98.90]
svm_score = [83.75, 90.30, 93.85, 96.05, 97.55, 98.40, 98.80, 99.05]
snr = [-6, -4, -2, 0, 2, 4, 6, 8]
lm_score = np.array(lm_score) * 0.01
nn_score = np.array(nn_score) * 0.01
nn_score2 = np.array(nn_score2) * 0.01
svm_score = np.array(svm_score) * 0.01
snr = np.array(snr)
ali_snr = np.linspace(-7, 10, 100)

# https://stackoverflow.com/questions/55725139/fit-sigmoid-function-s-shape-curve-to-data-using-python
def sigmoid(x, L ,x0, k):
    y = L / (1 + np.exp(-k*(x-x0)))
    return (y)

p0 = [max(lm_score), np.median(snr), 1] # this is an mandatory initial guess

popt, pcov = curve_fit(sigmoid, snr, lm_score, p0, method='dogbox')
print(popt)
plt.plot(ali_snr, sigmoid(ali_snr, *popt))
plt.scatter(snr, lm_score)

popt, pcov = curve_fit(sigmoid, snr, nn_score, p0, method='dogbox')
print(popt)
plt.plot(ali_snr, sigmoid(ali_snr, *popt))
plt.scatter(snr, nn_score)

popt, pcov = curve_fit(sigmoid, snr, nn_score2, p0, method='dogbox')
print(popt)
plt.plot(ali_snr, sigmoid(ali_snr, *popt))
plt.scatter(snr, nn_score2)

popt, pcov = curve_fit(sigmoid, snr, svm_score, p0, method='dogbox')
print(popt)
plt.plot(ali_snr, sigmoid(ali_snr, *popt))
plt.scatter(snr, svm_score)

plt.xlabel('SNR (dB)')
plt.ylabel('accuracy')
plt.legend(['lm', 'lm', 'nn old', 'nn old', 'nn new', 'nn new', 'svm', 'svm'])
plt.show()
