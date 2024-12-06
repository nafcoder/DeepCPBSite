import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import random
import csv
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score
from sklearn.preprocessing import PowerTransformer


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


feature_paths = {
    'ProtT5-XL-U50-train': '/media/nafiislam/T7/DeepCPBSite/all_features/ProtT5-XL-U50_training.csv',
    'ESM-2-train': '/media/nafiislam/T7/DeepCPBSite/all_features/ESM_2_training.csv',
    'Structural-train': '/media/nafiislam/T7/DeepCPBSite/all_features/structural_feature_training.csv',
    'word_embedding-train': '/media/nafiislam/T7/DeepCPBSite/all_features/word_embedding_training.csv',
}

# Set seeds for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Read training data
others = ['Structural-train', 'ESM-2-train', 'word_embedding-train']
file_path_Benchmark_embeddings = feature_paths['ProtT5-XL-U50-train']
D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=True)
feature_y_Benchmark_embeddings = D_feature.iloc[:, 0].values

feature_X_Benchmark_embeddings = np.zeros((feature_y_Benchmark_embeddings.shape[0], 1), dtype=float)
D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=True)
feature_X_Benchmark_embeddings = np.concatenate((feature_X_Benchmark_embeddings, D_feature.iloc[:, 1:].values), axis=1)

for other in others:
    file_path_Benchmark_embeddings = feature_paths[other]
    D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=True)
    if other == "Structural-train":
        feature_X_Benchmark_embeddings = np.concatenate((feature_X_Benchmark_embeddings, preprocess_the_dataset(D_feature.iloc[:, 1:].values)), axis=1)
    else:
        feature_X_Benchmark_embeddings = np.concatenate((feature_X_Benchmark_embeddings, D_feature.iloc[:, 1:].values), axis=1)

feature_X_Benchmark_embeddings = np.delete(feature_X_Benchmark_embeddings, 0, axis=1)

X = feature_X_Benchmark_embeddings
y = feature_y_Benchmark_embeddings

print(X.shape)
print(y.shape)

c = Counter(y)
print(c)

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


# Set parameters
input_size = X.shape[1]
batch_size = 2048

# Create DataLoader for training and testing
train_dataset = torch.utils.data.TensorDataset(torch.tensor(X).to(torch.float32), torch.tensor(y).to(torch.float32))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

X = None
y = None

model = BinaryClassifierLoss(input_size)

# Define loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

model.train()

# Training loop
for epoch in range(325):
    for i, (inputs, labels) in enumerate(train_dataloader):
        # Forward pass
        out, outputs = model(inputs)

        # Count positive and negative samples
        num_positive = (labels == 1).sum().item()  # Count positives
        num_negative = (labels == 0).sum().item()  # Count negatives

        # Calculate positive weight
        if num_positive > 0:  # Avoid division by zero
            pos_weight = num_negative / num_positive
        else:
            pos_weight = 1.0  # Default value if no positive samples

        # Define loss function with calculated positive weight
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))

        loss = criterion(out, labels.view(-1, 1))

        # Backward pass and optimization
        loss.backward()

        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Print gradient norm
        print("Gradient norm:", gradient_norm)

        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            print(f'Epoch [{epoch + 1}/{325}], Step [{i + 1}/{len(train_dataloader)}]')



print("Training finished.")
file = f'loss.pth'
torch.save(model, file)