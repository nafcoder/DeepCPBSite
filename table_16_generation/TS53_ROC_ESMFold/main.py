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

plt.plot([0.0, 1.0], [0.0, 1.0], color="darkblue", linestyle='--')

fprs = []
tprs = []
for model in models:
    file = f'./ROC/fpr_{model}.csv'
    with open(file, 'r') as f:
        fpr = np.loadtxt(f, delimiter=',').tolist()
        fprs.append(fpr)
    
    file = f'./ROC/tpr_{model}.csv'
    with open(file, 'r') as f:
        tpr = np.loadtxt(f, delimiter=',').tolist()
        tprs.append(tpr)

print(len(fprs))
print(len(tprs))

for i in range(len(models)):
    model = models[i]
    plt.plot(fprs[i], tprs[i], color=colors[i], label=model)

plt.title('Receiver operating characteristic curve (TS53 ESMFold)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.show()