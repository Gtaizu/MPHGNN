# Exploring ncRNA-drug sensitivity associations via graph contrastive learning
NDSGCL is a novel graph contrastive learning approach to predict ncRNA-drug sensitivity. NDSGCL uses graph convolutional networks to learn feature representations of ncRNAs and drugs in ncRNA-drug bipartite graphs. It integrates local structural neighbours and global semantic neighbours to learn a more comprehensive representation by contrastive learning. Specifically, the local structural neighbours aim to capture the higher-order relationship in the ncRNA-drug graph, while the global semantic neighbours are defined based on semantic clusters of the graph that can alleviate the impact of data sparsity.
# Requirements
- torch 1.8.0
- python 3.8.15
- faiss-gpu 1.7.2
- mkl 2021.4.0
- numpy 1.21.5
- scikit-learn 1.1.3
# Data
RNAactDrug is a database of RNA molecules(including mRNAs, lncRNAs, and miRNAs) and drug sensitivity associations, covering more than 19,770 mRNAs, 11,119 lncRNAs, 438 miRNAs, and 4,155 drugs. We obtained the lncRNA-disease association dataset {required for the experiment from this database}. To further verify the effectiveness of NDSGCL, we constructed three datasets according to the p-values of 0.05, 0.02, and 0.01. In addition, we also obtained the miRNA-drug sensitivity dataset from this database and fed this dataset into our model to verify the generality of our model. Through data cleaning methods such as removing redundancy, we finally obtained four benchmark datasets, in which LDA1 (p_value=0.05) contains 5,697,081 associations between 11,043 lncRNAs and 2,936 drugs, LDA2 (p_value=0.02) contains 544,355 associations between 11,031 lncRNAs and 2,936 drugs, LDA3 (p_value=0.01) contains 489,523 associations between 10,954 lncRNAs and 2,936 drugs and MDA contains 28,955 between 438 miRNAs and 1,995 drugs.

# Run the demo
```
python main.py
```
