import os
import torch as pt
from tqdm import tqdm
from glob import glob
import csv

from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

import numpy as np
import pandas as pd


def find_metrics(y_predict, y_proba, y_test):
    tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()  # y_true, y_pred

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    bal_acc = balanced_accuracy_score(y_test, y_predict)
    acc = accuracy_score(y_test, y_predict)

    if tp == 0 and fp == 0:
        prec = 0
    else:
        prec = tp / (tp + fp)

    if prec == 0 and sensitivity == 0:
        f1_score_1 = 0
    else:
        f1_score_1 = 2 * prec * sensitivity / (prec + sensitivity)
    mcc = matthews_corrcoef(y_test, y_predict)
    auc = roc_auc_score(y_test, y_proba)
    auPR = average_precision_score(y_test, y_proba)  # auPR

    return sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR


global_y_pred = []
global_y_proba = []
global_y_test = []


with open("TS53.fasta", 'r') as f:
    lines = f.readlines()

    for i in range(0, len(lines), 4):
        pdb_id = lines[i].strip()[1:]
        fasta = lines[i+1].strip()
        target = lines[i+3].strip().replace("-", "")

        # print(pdb_id, i // 4)
        # print(target)

        target = [int(value) for value in target]

        file = f'probabilities/{pdb_id}.txt'

        with open(file, 'r') as f:
            l = f.readlines()
            proba = [float(v.strip().split(' ')[1]) for v in l]
        

        if len(proba) != len(target):
            print("Lengths are not equal", pdb_id, len(proba), len(target))
            continue
        global_y_test.extend(target)
        global_y_proba.extend(proba)
        global_y_pred.extend([1 if value > 0.5 else 0 for value in proba])

global_y_pred =  np.array(global_y_pred)
global_y_proba = np.array(global_y_proba)
global_y_test = np.array(global_y_test)
print(global_y_pred.shape)
# print(global_y_pred)
print(global_y_proba.shape)
# print(global_y_proba)
print(global_y_test.shape)
# print(global_y_test)

sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR = find_metrics(
            global_y_pred, global_y_proba, global_y_test)

print(f"Sensitivity: {sensitivity:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"Balanced Accuracy: {bal_acc:.3f}")
print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"F1 Score: {f1_score_1:.3f}")
print(f"MCC: {mcc:.3f}")
print(f"AUC: {auc:.3f}")
print(f"AuPR: {auPR:.3f}")

fpr, tpr, _ = roc_curve(np.array(global_y_test), np.array(global_y_proba))

f = open("./outputs/fpr.csv", "w", newline="")
writer = csv.writer(f)
writer.writerows([[fp] for fp in fpr])
f.close()

f = open("./outputs/tpr.csv", "w", newline="")
writer = csv.writer(f)
writer.writerows([[tp] for tp in tpr])
f.close()

precision, recall, _ = precision_recall_curve(np.array(global_y_test), np.array(global_y_proba))

f_name = f'./outputs/precision.csv'
with open(f_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows([[p] for p in precision])

f_name = f'./outputs/recall.csv'
with open(f_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows([[r] for r in recall])

