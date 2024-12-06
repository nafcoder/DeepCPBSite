import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from matplotlib import pyplot as plt
import random
import pickle
import csv


plt.rcParams.update({'font.size': 12})

models =  ['DeepCPBSite', 'CAPSIF-G', 'CAPSIF-V', 'DeepGlycanSite']

cmap = plt.get_cmap('tab10')
colors = [cmap(i / (len(models) + 1)) for i in range(len(models))]

recall = []
precision = []
for model in models:
    file = f'./precision_recall_curve/recall_{model}.csv'
    with open(file, 'r') as f:
        re = np.loadtxt(f, delimiter=',').tolist()
        recall.append(re)
    
    file = f'./precision_recall_curve/precision_{model}.csv'
    with open(file, 'r') as f:
        pre = np.loadtxt(f, delimiter=',').tolist()
        precision.append(pre)

print(len(recall))
print(len(precision))

for i in range(len(models)):
    model = models[i]
    plt.plot(recall[i], precision[i], color=colors[i], label=model)

plt.title('Precision-Recall Curve (TS53 ESMFold)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='upper right')
plt.grid()
plt.show()

