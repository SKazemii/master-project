import numpy as np
from sklearn import metrics
from MLPackage.Deep_network import * 


y = np.array([0, 0, 1, 1, ])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
print('fpr:',fpr)

# print('tpr:', tpr)

print('1-tpr:', 1-tpr)

print('thresholds:', thresholds)
print()

FRR = list()
FAR = list()

y_pred = scores


fpr, tpr, threshold = roc_curve(y, scores, pos_label=1)
fnr = 1 - tpr
abs_diffs = np.abs(fpr - fnr)
min_index = np.argmin(abs_diffs)
breakpoint()
if metric=='eer':
    EER = np.mean((fpr[min_index], fnr[min_index]))
    best_thr = threshold[min_index]
if metric=='gmeans':
    gmeans = sqrt(tpr * (1-fpr));ix = argmax(gmeans)
    EER= gmeans[ix];
    best_thr=threshold[ix]
if metric=='zero_fpr':
    ix=np.max(np.where(fpr==0))
    best_thr=threshold[ix]
    EER=0
if metric=='zero_frr':
    ix=np.min(np.where(tpr==1))
    best_thr=threshold[ix]
    EER=0

print("FRR:", FRR)

print("FAR:", FAR)
print('thresholds:', np.linspace(0, 1, 5))


breakpoint()
