import random

from util import *
from NDSGCL import NDSGCL

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':

    conf = ModelConf('NDSGCL.conf')

    data = torch.load("MiDrug_test_data.pth", weights_only=False)
    # 提取 edge_label_index 和 edge_label
    edge_label_index = data[("miRNA", "MiDrug", "drug")].edge_label_index
    edge_label = data[("miRNA", "MiDrug", "drug")].edge_label
    edge_index = data[("miRNA", "MiDrug", "drug")].edge_index
    # 找到标签为 1 的边的索引位置
    pos_indices = (edge_label == 1).nonzero(as_tuple=True)[0]
    neg_indices = (edge_label == 0).nonzero(as_tuple=True)[0]
    # 提取对应的边（正样本边）
    pos_edge_index = edge_label_index[:, pos_indices]
    neg_edge_index = edge_label_index[:, neg_indices]
    train_edge_list = []
    test_edge_list = []

    for i in range(edge_index.size(1)):
        lncRNA = edge_index[0, i].item()
        drug = edge_index[1, i].item()
        train_edge_list.append([lncRNA, drug, 1.0])

    for i in range(pos_edge_index.size(1)):
        lncRNA = pos_edge_index[0, i].item()
        drug = pos_edge_index[1, i].item()
        test_edge_list.append([lncRNA, drug, 1.0])

    for i in range(neg_edge_index.size(1)):
        lncRNA = neg_edge_index[0, i].item()
        drug = neg_edge_index[1, i].item()
        test_edge_list.append([lncRNA, drug, 0])

    training_data = train_edge_list
    test_data = test_edge_list
    model = NDSGCL(conf=conf, training_set=training_data, test_set=test_data, i=i)
    model.train()  # 用实例来调用 train 方法


