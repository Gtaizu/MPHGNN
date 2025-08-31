# MPHGNN
Description
MicroRNAs (miRNAs) play a crucial regulatory role in gene expression and are closely associated with drug resistance in various diseases, especially cancer. Accurately predicting miRNA-drug resistance associations is vital for understanding resistance mechanisms and developing effective therapeutics.

We developed a novel framework, MPHGNN, for predicting miRNA-drug resistance associations. MPHGNN constructs a heterogeneous network integrating miRNAs, genes, and drugs, incorporating similarity features and biological interactions. To learn deep representations from this network, we designed a meta-path guided heterogeneous graph neural network architecture. The model effectively captures complex, higher-order biological relationships by leveraging semantically rich meta-paths. Finally, a joint predictor with interpretability components is designed to achieve accurate prediction and provide biological insights into miRNA-drug resistance mechanisms.

# Availability

Datasets and source code are available at: https://github.com/Gtaizu/MPHGNN

# File Description:

MiDrug_data_end.pth: miRNA-drug resistance association dataset
# Local running
# Environment
Before running, please make sure the following packages are installed in your Python environment. We strongly recommend using a virtual environment.

Core Dependencies:

python=3.10

torch=2.1.0+cu118

dgl=2.2.1+cu118

torch-geometric=2.6.1

Basic Data Handling & Utilities:

pandas>=1.5.0

numpy>=1.24.0

scikit-learn>=1.2.0

tqdm>=4.65.0

matplotlib>=3.7.0

seaborn>=0.12.0
