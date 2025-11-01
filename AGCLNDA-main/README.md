# AGCLNDA
Non-coding RNAs (ncRNAs), which do not
encode proteins, have been implicated in chemotherapy resistance in cancer treatment. Given the high costs and time
required for traditional biological experiments, there is a
pressing need for computational models to predict ncRNAdrug resistance associations. In this study, we introduce
AGCLNDA, an adaptive contrastive learning method designed to uncover these associations. AGCLNDA begins
by constructing a bipartite graph from existing ncRNA-drug
resistance data. It then utilizes a light graph convolutional
network (LightGCN) to learn vector representations for both
ncRNAs and drugs. The method assesses resistance association scores through the inner product of these vectors.
To tackle data sparsity and noise, AGCLNDA incorporates
learnable augmented view generators and denoised view
generators, which provide contrastive views for enhanced
data augmentation. Comparative experiments demonstrate
that AGCLNDA outperforms five other advanced methods. Case studies further validate AGCLNDA as an effective tool for predicting ncRNA-drug resistance associations.
# Requirements
- torch 1.11.0
- python 3.9.19
- numpy 1.26.4
- pandas 2.2.2
- scikit-learn 1.4.2
- scipy 1.13.0

# Data
NoncoRNA [40] is an experimentally supported database
containing 5,568 ncRNAs and 154 drugs across 134 cancers. We used the publicly released version from February
2020, available at http://www.ncdtcdb.cn:8080/NoncoRNA.
ncRNADrug [41] is a comprehensive database with 29,551
experimentally validated ncRNA-drug resistance associations involving 9,195 ncRNAs and 266 drugs. We
utilized the August 2023 update, downloadable from
http://www.jianglab.cn/ncRNADrug. Merging these datasets
and removing redundancies, we obtained 9,633 unique
ncRNA-drug resistance association pairs for 912 ncRNAs and 374 drugs.

# Project structure
Datasets/mydata1: dataset

model.py: model related code

DataHandler.py:data related code

main.py: main function

# Run the demo
```
python main.py
```
