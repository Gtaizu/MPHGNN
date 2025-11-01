import random

import torch

data = torch.load("LncDrug_test_data.pth", weights_only=False)
# 提取 edge_label_index 和 edge_label
edge_label_index = data[("lncRNA", "LncDrug", "drug")].edge_label_index
edge_label = data[("lncRNA", "LncDrug", "drug")].edge_label

# 找到标签为 1 的边的索引位置
pos_indices = (edge_label == 1).nonzero(as_tuple=True)[0]

# 提取对应的边（正样本边）
pos_edge_index = edge_label_index[:, pos_indices]

# 设置随机种子以复现结果
random.seed(42)

# 获取边总数
num_edges = pos_edge_index.size(1)
indices = list(range(num_edges))
random.shuffle(indices)

# 划分索引
train_size = int(0.8 * num_edges)
train_idx = indices[:train_size]
test_idx = indices[train_size:]

# 构建训练集和测试集的 pos_edge_index
train_pos_edge_index = pos_edge_index[:, train_idx]
test_pos_edge_index = pos_edge_index[:, test_idx]

train_edge_list = []
test_edge_list = []

for i in range(train_pos_edge_index.size(1)):
    lncRNA = train_pos_edge_index[0, i].item()
    drug = train_pos_edge_index[1, i].item()
    train_edge_list.append([lncRNA, drug, 1.0])

for i in range(test_pos_edge_index.size(1)):
    lncRNA = test_pos_edge_index[0, i].item()
    drug = test_pos_edge_index[1, i].item()
    test_edge_list.append([lncRNA, drug, 1.0])

# training_data = train_edge_list
# test_data = test_edge_list
# model = NDSGCL(conf=conf, training_set=training_data, test_set=test_data, i=i)
# model.train()  # 用实例来调用 train 方法