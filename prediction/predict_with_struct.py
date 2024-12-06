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



X = pd.read_csv('features_with_struct.csv', header=None, low_memory=True).values
print(X.shape)

X_test = torch.tensor(X, dtype=torch.float32)

y_predict, y_proba = EnsembleModel(
    torch.load('undersample.pth'),
    torch.load('loss.pth'),
    torch.load('weighted.pth')
).predict(X_test)

y_predict = y_predict.detach().numpy()
y_proba = y_proba.detach().numpy()

with open('prediction.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['prediction', 'probability'])
    for i in range(len(y_predict)):
        writer.writerow([y_predict[i][0], y_proba[i][0]])
    