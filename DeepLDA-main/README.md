# Identification of associations between lncRNA and drug resistance based on deep learning and attention mechanism (DeepLDA)

## Requirements
DeepLDA is tested in the conda environment. Note: Before using DeepLDA, users first need to prepare the following softwares in the execution environment：
  * Python 3.6.2
  * PyTrorch 1.9.1
  * NumPy 1.19.2
  * Scipy 1.5.2
  * scikit-learn 0.24.1

## Code
This directory stores the python code of the model
  * main_gpu.py
  >It is used to compute the performance of the model for lncRNA-drug resistance association prediction
  * models.py
  >It is used to define the model.
  * layers.py
  >It is used to define GAT and GCN layer.
  * utils.py
  >It is used to define the functions that need to be used in the model.

## Usage
Note: Go to the /Deep-LDA/Code/ directory before using this model.
  * Please run the following python command：```python main_cpu.py```
  
## Datasets
This directory stores the datasets used by the model
### Dataset1: NoncoRNA dataset
  * lnc_name.txt
  > Names of lncRNAs
  * drug_name.txt
  > Names of drugs
  * lnc_drug_net.txt
  > LncRNA-drug resistance associations
  * lnc_drug_net.csv
  > LncRNA-drug resistance associations
### Dataset2: ncDR dataset
  * lnc_name.txt
  > Names of lncRNAs
  * drug_name.txt
  > Names of drugs
  * lnc_drug_net.txt
  > LncRNA-drug resistance associations
  * lnc_drug_net.csv
  > LncRNA-drug resistance associations

## Results
This directory stores the results of the model
### 1: Result of NoncoRNA dataset
### 2: Result of ncDR dataset

