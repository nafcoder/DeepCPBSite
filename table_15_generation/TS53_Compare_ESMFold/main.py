import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter
import random
import csv
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score
from sklearn.preprocessing import PowerTransformer
import csv
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve


def preprocess_the_dataset(feature_X):

    pt = PowerTransformer()
    pt.fit(feature_X)
    feature_X = pt.transform(feature_X)

    return feature_X


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

def getFormat(metric):
    return "{:.3f}".format(metric)

# Residual block with Conv1D
class ResidualBlock(nn.Module):
    def __init__(self, filters, input_channels=21):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=3, padding=1)
    
    def forward(self, x):
        residual = x
        out = torch.nn.functional.relu(self.conv1(x))
        out = torch.nn.functional.relu(self.conv2(out))
        out = torch.cat((out, residual), dim=1)  # Concatenate along the channel dimension
        return out


# Complete ResNet + ANN model
class BinaryClassifierUndersample(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifierUndersample, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=21, embedding_dim=21)
        
        # Conv1D branch for sequence data
        self.res_block1 = ResidualBlock(10)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.res_block2 = ResidualBlock(10, input_channels=31)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.res_block3 = ResidualBlock(10, input_channels=41)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        
        # Fully connected layers for Conv1D branch
        self.fc_conv1 = nn.Linear(51*3, 32)  # After two MaxPooling1D
        self.bn_conv1 = nn.BatchNorm1d(32)
        self.relu_conv1 = nn.ReLU()
        self.dropout_conv = nn.Dropout(0.5)
        
        #ann branch
        self.fc1 = nn.Linear(input_size - 31, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        
        # Combined output layers
        self.fc_combined1 = nn.Linear(256+32, 128)
        self.bn_combined1 = nn.BatchNorm1d(128)
        self.relu_combined1 = nn.ReLU()
        self.dropout_combined = nn.Dropout(0.5)

        self.fc_output = nn.Linear(128, 1)

    def forward(self, x):
        x_conv = x[:, -31:].int()
        x_ann = x[:, :-31]

        # Conv1D branch
        x_conv = self.embedding(x_conv)
        x_conv = x_conv.permute(0, 2, 1)
        x_conv = self.res_block1(x_conv)
        x_conv = self.pool1(x_conv)
        x_conv = self.res_block2(x_conv)
        x_conv = self.pool2(x_conv)
        x_conv = self.res_block3(x_conv)
        x_conv = self.pool3(x_conv)
        x_conv = self.flatten(x_conv)

        x_conv = self.fc_conv1(x_conv)
        x_conv = self.bn_conv1(x_conv)
        x_conv = self.relu_conv1(x_conv)
        x_conv = self.dropout_conv(x_conv)

        # ANN branch
        x_ann = self.fc1(x_ann)
        x_ann = self.bn1(x_ann)
        x_ann = self.relu1(x_ann)
        x_ann = self.dropout1(x_ann)

        x_ann = self.fc2(x_ann)
        x_ann = self.bn2(x_ann)
        x_ann = self.relu2(x_ann)
        x_ann = self.dropout2(x_ann)

        # Concatenate the outputs of the two branches
        x_combined = torch.cat((x_conv, x_ann), dim=1)

        x_combined = self.fc_combined1(x_combined)
        x_combined = self.bn_combined1(x_combined)
        x_combined = self.relu_combined1(x_combined)
        x_combined = self.dropout_combined(x_combined)

        output = self.fc_output(x_combined)
        output = torch.sigmoid(output)
        
        return output
    

# Complete ResNet + ANN model
class BinaryClassifierWeighted(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifierWeighted, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=21, embedding_dim=21)
        
        # Conv1D branch for sequence data
        self.res_block1 = ResidualBlock(10)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.res_block2 = ResidualBlock(10, input_channels=31)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.res_block3 = ResidualBlock(10, input_channels=41)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        
        # Fully connected layers for Conv1D branch
        self.fc_conv1 = nn.Linear(51*3, 32)  # After two MaxPooling1D
        self.bn_conv1 = nn.BatchNorm1d(32)
        self.relu_conv1 = nn.ReLU()
        self.dropout_conv = nn.Dropout(0.5)
        
        #ann branch
        self.fc1 = nn.Linear(input_size - 31, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        
        # Combined output layers
        self.fc_combined1 = nn.Linear(256+32, 128)
        self.bn_combined1 = nn.BatchNorm1d(128)
        self.relu_combined1 = nn.ReLU()
        self.dropout_combined = nn.Dropout(0.5)

        self.fc_output = nn.Linear(128, 1)

    def forward(self, x):
        x_conv = x[:, -31:].int()
        x_ann = x[:, :-31]

        # Conv1D branch
        x_conv = self.embedding(x_conv)
        x_conv = x_conv.permute(0, 2, 1)
        x_conv = self.res_block1(x_conv)
        x_conv = self.pool1(x_conv)
        x_conv = self.res_block2(x_conv)
        x_conv = self.pool2(x_conv)
        x_conv = self.res_block3(x_conv)
        x_conv = self.pool3(x_conv)
        x_conv = self.flatten(x_conv)

        x_conv = self.fc_conv1(x_conv)
        x_conv = self.bn_conv1(x_conv)
        x_conv = self.relu_conv1(x_conv)
        x_conv = self.dropout_conv(x_conv)

        # ANN branch
        x_ann = self.fc1(x_ann)
        x_ann = self.bn1(x_ann)
        x_ann = self.relu1(x_ann)
        x_ann = self.dropout1(x_ann)

        x_ann = self.fc2(x_ann)
        x_ann = self.bn2(x_ann)
        x_ann = self.relu2(x_ann)
        x_ann = self.dropout2(x_ann)

        # Concatenate the outputs of the two branches
        x_combined = torch.cat((x_conv, x_ann), dim=1)

        x_combined = self.fc_combined1(x_combined)
        x_combined = self.bn_combined1(x_combined)
        x_combined = self.relu_combined1(x_combined)
        x_combined = self.dropout_combined(x_combined)

        output = self.fc_output(x_combined)
        output = torch.sigmoid(output)
        
        return output


# Complete ResNet + ANN model
class BinaryClassifierLoss(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifierLoss, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=21, embedding_dim=21)
        
        # Conv1D branch for sequence data
        self.res_block1 = ResidualBlock(10)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.res_block2 = ResidualBlock(10, input_channels=31)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.res_block3 = ResidualBlock(10, input_channels=41)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        
        # Fully connected layers for Conv1D branch
        self.fc_conv1 = nn.Linear(51*3, 32)  # After two MaxPooling1D
        self.bn_conv1 = nn.BatchNorm1d(32)
        self.relu_conv1 = nn.ReLU()
        self.dropout_conv = nn.Dropout(0.5)
        
        #ann branch
        self.fc1 = nn.Linear(input_size - 31, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        
        # Combined output layers
        self.fc_combined1 = nn.Linear(256+32, 128)
        self.bn_combined1 = nn.BatchNorm1d(128)
        self.relu_combined1 = nn.ReLU()
        self.dropout_combined = nn.Dropout(0.5)

        self.fc_output = nn.Linear(128, 1)

    def forward(self, x):
        x_conv = x[:, -31:].int()
        x_ann = x[:, :-31]

        # Conv1D branch
        x_conv = self.embedding(x_conv)
        x_conv = x_conv.permute(0, 2, 1)
        x_conv = self.res_block1(x_conv)
        x_conv = self.pool1(x_conv)
        x_conv = self.res_block2(x_conv)
        x_conv = self.pool2(x_conv)
        x_conv = self.res_block3(x_conv)
        x_conv = self.pool3(x_conv)
        x_conv = self.flatten(x_conv)

        x_conv = self.fc_conv1(x_conv)
        x_conv = self.bn_conv1(x_conv)
        x_conv = self.relu_conv1(x_conv)
        x_conv = self.dropout_conv(x_conv)

        # ANN branch
        x_ann = self.fc1(x_ann)
        x_ann = self.bn1(x_ann)
        x_ann = self.relu1(x_ann)
        x_ann = self.dropout1(x_ann)

        x_ann = self.fc2(x_ann)
        x_ann = self.bn2(x_ann)
        x_ann = self.relu2(x_ann)
        x_ann = self.dropout2(x_ann)

        # Concatenate the outputs of the two branches
        x_combined = torch.cat((x_conv, x_ann), dim=1)

        x_combined = self.fc_combined1(x_combined)
        x_combined = self.bn_combined1(x_combined)
        x_combined = self.relu_combined1(x_combined)
        x_combined = self.dropout_combined(x_combined)

        output = self.fc_output(x_combined)

        return output, torch.sigmoid(output)


class EnsembleModel:
    def __init__(self, undersample_model, loss_model, weighted_model):
        self.undersample_model = undersample_model
        self.loss_model = loss_model
        self.weighted_model = weighted_model
    
    def predict(self, x):
        self.undersample_model.eval()
        self.loss_model.eval()
        self.weighted_model.eval()

        with torch.no_grad():
            self.y_proba_undersample = self.undersample_model(x).detach()
            self.y_proba_loss = self.loss_model(x)[1].detach()
            self.y_proba_weighted = self.weighted_model(x).detach()

            self.y_proba = torch.mean(torch.stack([self.y_proba_undersample, self.y_proba_undersample, self.y_proba_loss, self.y_proba_weighted]), dim=0)

            self.y_predict = (self.y_proba > 0.5).float()

        return self.y_predict, self.y_proba


# Define file paths
feature_paths = {
    'ProtT5-XL-U50-test': '/media/nafiislam/T7/DeepCPBSite/all_features/TS53/ProtT5-XL-U50_independent.csv',
    'ESM-2-test': '/media/nafiislam/T7/DeepCPBSite/all_features/TS53/ESM_2_independent.csv',
    'Structural-test': '/media/nafiislam/T7/DeepCPBSite/all_features/ESMFold/TS53/structural_feature_independent.csv',
    'word_embedding-test': '/media/nafiislam/T7/DeepCPBSite/all_features/TS53/word_embedding_independent.csv',
}

# Set seeds for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Read training data
others = ['Structural-test', 'ESM-2-test', 'word_embedding-test']
file_path_Benchmark_embeddings = feature_paths['ProtT5-XL-U50-test']
D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=True)
feature_y_Benchmark_embeddings = D_feature.iloc[:, 0].values

feature_X_Benchmark_embeddings = np.zeros((feature_y_Benchmark_embeddings.shape[0], 1), dtype=float)
D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=True)
feature_X_Benchmark_embeddings = np.concatenate((feature_X_Benchmark_embeddings, D_feature.iloc[:, 1:].values), axis=1)

for other in others:
    file_path_Benchmark_embeddings = feature_paths[other]
    D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=True)
    if other == "Structural-test":
        feature_X_Benchmark_embeddings = np.concatenate((feature_X_Benchmark_embeddings, preprocess_the_dataset(D_feature.iloc[:, 1:].values)), axis=1)
    else:
        feature_X_Benchmark_embeddings = np.concatenate((feature_X_Benchmark_embeddings, D_feature.iloc[:, 1:].values), axis=1)

feature_X_Benchmark_embeddings = np.delete(feature_X_Benchmark_embeddings, 0, axis=1)

X = feature_X_Benchmark_embeddings
y = feature_y_Benchmark_embeddings

c = Counter(y)
print(c)

X_test = torch.tensor(X, dtype=torch.float32)
y_test = torch.tensor(y, dtype=torch.float32)

y_predict, y_proba = EnsembleModel(
    torch.load('undersample.pth'),
    torch.load('loss.pth'),
    torch.load('weighted.pth')
).predict(X_test)

y_predict = y_predict.detach().numpy()
y_proba = y_proba.detach().numpy()

sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR = find_metrics(y_predict, y_proba, y_test)

writer = csv.writer(open('metrics.csv', 'w'))
writer.writerow(['sensitivity', 'specificity', 'bal_acc', 'acc', 'prec', 'f1_score_1', 'mcc', 'auc', 'auPR'])

writer.writerow([getFormat(sensitivity), getFormat(specificity), getFormat(bal_acc), getFormat(acc), getFormat(prec), getFormat(f1_score_1), getFormat(mcc), getFormat(auc), getFormat(auPR)])

fpr, tpr, _ = roc_curve(np.array(y_test), np.array(y_proba))

f = open("./outputs/fpr.csv", "w", newline="")
writer = csv.writer(f)
writer.writerows([[fp] for fp in fpr])
f.close()

f = open("./outputs/tpr.csv", "w", newline="")
writer = csv.writer(f)
writer.writerows([[tp] for tp in tpr])
f.close()

precision, recall, _ = precision_recall_curve(np.array(y_test), np.array(y_proba))

f_name = f'./outputs/precision.csv'
with open(f_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows([[p] for p in precision])

f_name = f'./outputs/recall.csv'
with open(f_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows([[r] for r in recall])

