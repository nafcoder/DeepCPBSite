# Predicting Protein-Carbohydrate Binding Sites: A Deep Learning Approach Integrating Protein Language Model Embeddings and Structural Features

Here, we have built a novel ensemble model, DeepCPBSite, combined with three separate models (Random undersampling, Weighted oversampling and Class-Weighted Loss) based on ResNet+FNN deep learning architecture. The framework for this architecture is given below:

![DeepCPBSite-1](https://github.com/user-attachments/assets/2183d2f4-20ca-47b1-8615-4dc688dbe649)

![DeepCPBSite_2-1](https://github.com/user-attachments/assets/0a299943-6754-4744-afe4-3f96e7a1179d)



# Data availability
Training set, independent set, TS53 set and TS37 set are given in [Dataset](Dataset) folder.

# Environments
OS: Pop!_OS 22.04 LTS

Python version: Python 3.9.19


Used libraries: 
```
numpy==1.26.4
pandas==2.2.1
pytorch==2.4.1
xgboost==2.0.3
pickle5==0.0.11
scikit-learn==1.2.2
matplotlib==3.8.2
PyQt5==5.15.10
imblearn==0.0
skops==0.9.0
shap==0.45.1
IPython==8.18.1
tqdm==4.66.5
biopython==1.84
transformers==4.44.2
```

### Reproduction of results
1. Firstly, download all features. Read the readme.txt of  [all_features](all_features) folder

2. All reproducable codes are given. Training and prediction scripts are also provided.

3. For reproducing results of each table inside the paper, you can navigate to the generation folder of that corresponding table number. Before running, update the feature_path variables inside the python files.

### Prediction
#### Prerequisites
1. transformers and Pytorch are needed for extracting the embeddings.

2. For more query, you can visit the following githubs:

    [ProtT5-XL-U50](https://github.com/agemagician/ProtTrans)

    [ESM2](https://github.com/facebookresearch/esm)

3. You need to install dssp for generating the structural features from PDB

```
sudo apt-get install dssp
```

#### Steps
1. Firsly, you need to fillup [dataset.txt](prediction/dataset.txt). Follow the pattern shown below:

```
>Protein_id
Fasta
```

2. For predicting carbohydrate protein binding sites from a protein sequence, you need to run the [extractFeatures.py](prediction/extractFeatures.py) to generate features and then run [predict_with_struct.py](prediction/predict_with_struct.py) for prediction with struct or [predict_without_struct.py](prediction/predict_without_struct.py) for prediction without struct.

3. For running [predict_with_struct.py](prediction/predict_with_struct.py), you need to input the PDB file for the query protein sequence. For generating ESMFold or AlphaFold PDB, you can visit: [ColabFold](https://github.com/sokrypton/ColabFold).

### Reproduce previous paper metrics
In [Prev_Papers](table_15_generation/Prev_Papers) and [Prev_Papers_ESMFold](table_16_generation/Prev_Papers_ESMFold), scripts are provided for reproducing the results of previous papers. We have given the probabilities that were produced from their scripts for TS53 set.
