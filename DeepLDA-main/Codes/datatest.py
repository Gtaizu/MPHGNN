import argparse

import torch


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run GAT-LGA.")
    parser.add_argument("--dataset_path",nargs="?",default="../Datasets",help="Training datasets.")#Dataset1,Dataset2
    parser.add_argument("--result_path",nargs="?",default="../Results/2/",help="Training datasets.")#1,2
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1050, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--nhid', type=int, default=50, help='Number of hidden units.')
    parser.add_argument('--nclass', type=int, default=200, help='Number of hidden units.')
    parser.add_argument('--nheads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=1000, help='Patience')
    return parser.parse_args()


args = parameter_parser()

data = torch.load("LncDrug_test_data.pth", weights_only=False)
edge_index = data[("lncRNA", "LncDrug", "drug")].edge_index

# 提取 edge_label_index 和 edge_label
edge_label_index = data[("lncRNA", "LncDrug", "drug")].edge_label_index
edge_label = data[("lncRNA", "LncDrug", "drug")].edge_label

# 找到标签为 1 的边的索引位置
pos_indices = (edge_label == 1).nonzero(as_tuple=True)[0]

# 提取对应的边（正样本边）
pos_edge_index = edge_label_index[:, pos_indices]
# print(edge_label_index)
# print(pos_edge_index)
# 获取 lncRNA 和 drug 的节点数

# 获取所有涉及的 lncRNA 和 drug 节点
lncRNA_nodes = pos_edge_index[0].unique()
drug_nodes = pos_edge_index[1].unique()

# 重新编号映射：旧索引 -> 新索引（0~N-1）
lncRNA_map = {old.item(): new for new, old in enumerate(lncRNA_nodes)}
drug_map = {old.item(): new for new, old in enumerate(drug_nodes)}

# 应用映射：构建新索引的 edge 列表
new_lncRNA_idx = torch.tensor([lncRNA_map[i.item()] for i in pos_edge_index[0]])
new_drug_idx = torch.tensor([drug_map[i.item()] for i in pos_edge_index[1]])

# 构建稠密关联矩阵
adj_matrix = torch.zeros((len(lncRNA_nodes), len(drug_nodes)), dtype=torch.float32)
adj_matrix[new_lncRNA_idx, new_drug_idx] = 1.0
print(adj_matrix)
print(adj_matrix.shape)
num_lnc = data["lncRNA"].num_nodes
num_drug = data["drug"].num_nodes

# 初始化邻接矩阵，shape = [num_lnc, num_drug]
adj_matrix = torch.zeros((num_lnc, num_drug), dtype=torch.float32)

# edge_index[0] 是 lncRNA 索引，edge_index[1] 是 drug 索引
adj_matrix[pos_edge_index[0], pos_edge_index[1]] = 1.0

LG_matrix = adj_matrix.cpu().numpy()

# 打乱列（即边的顺序）
num_edges = LG_matrix.shape[1]
perm = torch.randperm(num_edges)  # 生成一个打乱顺序的 index
LG_matrix = LG_matrix[:, perm]

print(data)
print(LG_matrix)

#print(data)