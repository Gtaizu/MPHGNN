from itertools import combinations
import torch.nn.functional as F
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
from torch import nn, Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import to_undirected, sort_edge_index, degree
from torch_geometric.utils.num_nodes import maybe_num_nodes
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
from torch_geometric.data import HeteroData


class GCN_GAT_GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        # 第一层 GCN
        self.gcn1 = GCNConv(in_channels, hidden_channels)

        # 第二层 GAT
        self.gat = GATConv(hidden_channels, hidden_channels, heads=heads, concat=False, edge_dim=1)

        # 第三层 GCN
        self.gcn2 = GCNConv(hidden_channels, out_channels)

        # cnn
        self.cnn = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)

    def forward(self, x, edge_index):
        # 第1层 GCN
        x1 = self.gcn1(x, edge_index)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        # 第2层 GAT
        x2 = self.gat(x1, edge_index)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        # 第3层 GCN
        x3 = self.gcn2(x2, edge_index)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, p=self.dropout, training=self.training)

        # 拼接两个嵌入
        x = torch.cat((x1, x3), dim=1)

        x = x.T.unsqueeze(0)

        x = self.cnn(x)
        x = x.squeeze(0).T

        print(x.shape)
        return x


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def draw_auc(y, pred, l):
    fpr, tpr, _ = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='{}:AUC = %0.4f'.format(l) % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)


def draw_aupr(y, pred, l):
    average_precision = average_precision_score(y, pred)
    precision, recall, _ = precision_recall_curve(y, pred)
    plt.plot(recall, precision, label='{}:AUPR = %0.4f'.format(l) % average_precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUPR Curve')
    plt.legend(loc='lower right')
    plt.grid(True)


def print_result(result):
    metrics = ['auc', 'aupr', 'acc', 'sen', 'pre', 'spe', 'F1', 'mcc']
    metric_values = [[] for _ in range(len(metrics))]
    for i in result:
        for j, val in enumerate(i):
            metric_values[j].append(val)
    metric_values = [np.array(m) for m in metric_values]
    formatted_metrics = []
    for metric, values in zip(metrics, metric_values):
        mean = "{:.4f}".format(values.mean())
        std = "{:.4f}".format(np.std(values))
        formatted_metrics.append(f"{metric}: {mean} ± {std}")
    print(*formatted_metrics)


# 边掩码策略
# edge_index	图的边连接（形状 [2, num_edges]）
def mask_path(edge_index, p, walks_per_node, walk_length, num_nodes):
    # 初始化边掩码为全部为True
    edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
    #  计算节点数
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    # 通过 COO 格式对边进行排序，以支持后面构建 CSR 格式邻接表。
    # row 是边的起始节点数组，col 是终止节点数组
    edge_index = sort_edge_index(edge_index, num_nodes=num_nodes)
    row, col = edge_index
    # 以概率 p 从 row 中采样一些边作为掩码起点（实际上是起始节点）
    # 然后每个起点复制 walks_per_node 次，得到随机游走的起点列表 start
    sample_mask = torch.rand(row.size(0), device=edge_index.device) <= p
    start = row[sample_mask].repeat(walks_per_node)
    # deg: 每个节点的出度（边数量）
    # rowptr: CSR格式的起点索引数组，标识每个节点邻接边的位置范围
    deg = degree(row, num_nodes=num_nodes)
    rowptr = row.new_zeros(num_nodes + 1)
    torch.cumsum(deg, 0, out=rowptr[1:])
    # 从每个起点开始执行随机游走，走 walk_length 步，e_id 是游走路径中用到的边的编号
    n_id, e_id = torch.ops.torch_cluster.random_walk(rowptr, col, start, walk_length, 1.0, 1.0)
    # 去除无效的边索引
    e_id = e_id[e_id != -1].view(-1)
    #  掩盖这些边
    edge_mask[e_id] = False
    # 第一部分：保留的边 → 用作图结构输入
    # 第二部分：掩盖的边 → 用作预测目标
    return edge_index[:, edge_mask], edge_index[:, ~edge_mask]


class MaskPath(torch.nn.Module):
    def __init__(self, p, walk_length, num_nodes):
        super(MaskPath, self).__init__()
        self.p = p  # 掩码概率（随机游走起始点的比例）
        self.walk_length = walk_length  # 随机游走长度
        self.num_nodes = num_nodes  # 节点总数

    def forward(self, edge_index):
        remaining_edges, masked_edges = mask_path(edge_index, self.p, 1, self.walk_length, self.num_nodes)
        remaining_edges = to_undirected(remaining_edges)
        return remaining_edges, masked_edges


def calculate_metrics(y_true, y_pred):
    TP = sum((y_true[i] == 1 and y_pred[i] == 1) for i in range(len(y_true)))
    TN = sum((y_true[i] == 0 and y_pred[i] == 0) for i in range(len(y_true)))
    FP = sum((y_true[i] == 0 and y_pred[i] == 1) for i in range(len(y_true)))
    FN = sum((y_true[i] == 1 and y_pred[i] == 0) for i in range(len(y_true)))

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-10)
    sensitivity = TP / (TP + FN + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    specificity = TN / (TN + FP + 1e-10)
    mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    F1_score = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-10)
    return accuracy, sensitivity, precision, specificity, F1_score, mcc


# 构造 miRNA-drug 关联图 数据, 对其进行训练/测试集的划分
def get_data():
    print("Loading data.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load("data/ncRNADrug3.pth", weights_only=False)
    train_edge_index = data['ncRNA', 'ncDrug', 'drug'].edge_index.to(device)
    train_edge_label = torch.ones(train_edge_index.size(1), dtype=torch.float32, device=device)

    num_miRNA = data['ncRNA'].num_nodes
    # 将 drug 节点编号偏移
    edge_index_adjusted = train_edge_index.clone()
    edge_index_adjusted[1] += num_miRNA

    miRNA_features = data['ncRNA'].x.to(device)
    drug_features = data['drug'].x.to(device)

    # 假设我们在 DataLoader 中或初始化时处理
    linear_proj = torch.nn.Linear(768, 938).to(device)
    linear_proj.eval()
    with torch.no_grad():  # 关闭梯度计算，节省显存
        mapped_drug_features = linear_proj(drug_features)

    # 然后拼接
    combined_features = torch.cat([miRNA_features, mapped_drug_features], dim=0)

    # 获取 edge_label 和 edge_label_index
    label = data['ncRNA', 'ncDrug', 'drug'].edge_label
    label_index = data['ncRNA', 'ncDrug', 'drug'].edge_label_index
    label_index[1] += num_miRNA

    # 找到正负样本的索引位置
    pos_mask = label == 1
    neg_mask = label == 0

    # 分离正负样本的边索引
    pos_edge_index = label_index[:, pos_mask]  # [2, num_positive]
    neg_edge_index = label_index[:, neg_mask]  # [2, num_negative]

    # 分离对应的标签（可选）
    pos_edge_label = label[pos_mask]  # 全是 1
    neg_edge_label = label[neg_mask]  # 全是 0

    # # 获取总数
    # num_pos = pos_edge_index.shape[1]
    # train_size = int(num_pos * 0.8)
    # test_size = num_pos - train_size
    #
    # # 打乱索引顺序
    # perm = torch.randperm(num_pos)
    # shuffled_edges = pos_edge_index[:, perm]
    # shuffled_labels = pos_edge_label[perm]
    #
    # # 划分训练集和测试集
    # train_pos_edge_index = shuffled_edges[:, :train_size]
    # test_pos_edge_index = shuffled_edges[:, train_size:]
    #
    # train_pos_edge_label = shuffled_labels[:train_size]  # 全为 1
    # test_pos_edge_label = shuffled_labels[train_size:]  # 全为 1
    # 负样本总数
    num_neg = neg_edge_index.shape[1]

    # 目标负样本数量 = 测试集正样本数量
    target_neg_num = pos_edge_index.shape[1]
    #
    # 负样本随机采样索引
    neg_perm = torch.randperm(num_neg)[:target_neg_num]

    # 采样后的负样本边索引和标签
    sampled_neg_edge_index = neg_edge_index[:, neg_perm]
    sampled_neg_edge_label = neg_edge_label[neg_perm]

    # 你可以把采样后的负样本用于测试集
    test_neg_edge_index = sampled_neg_edge_index
    test_neg_edge_label = sampled_neg_edge_label

    # 原始 pos_edge_index: shape [2, num_edges]
    # 生成反向边
    reverse_edge_index = edge_index_adjusted[[1, 0], :]  # 交换 i, j -> j, i

    # 拼接正向边和反向边
    bidirected_pos_edge_index = torch.cat([edge_index_adjusted, reverse_edge_index], dim=1)

    train_data = Data(
        x=combined_features,  # 节点特征矩阵
        edge_index=bidirected_pos_edge_index,  # 边索引矩阵
        pos_edge_label=train_edge_label,  # 训练集正样本标签（通常都是1）
        pos_edge_label_index=edge_index_adjusted,  # 训练集正样本边索引
    )

    test_data = Data(
        x=combined_features,  # 节点特征矩阵
        edge_index=bidirected_pos_edge_index,  # 边索引矩阵
        pos_edge_label=pos_edge_label,  # 训练集正样本标签（通常都是1）
        pos_edge_label_index=pos_edge_index,  # 训练集正样本边索引
        neg_edge_label=neg_edge_label,
        neg_edge_label_index=neg_edge_index
    )

    splits = dict(train=train_data, test=test_data)
    return splits

    # miRNA_drug = pd.read_csv("data/data_3000.csv", header=None)
    # miRNA_list = list(set(miRNA_drug[0]))  # 701
    # drug_list = list(set(miRNA_drug[1]))  # 101
    # # 将 miRNA 和 drug 统一编码成索引。
    # # miRNA 从 0 开始，drug 的索引从 len(miRNA_list) 开始，确保节点不重合。
    # # 构造出边索引矩阵 adj，形状是 [2, num_edges]，符合 PyG 格式
    # adj = torch.LongTensor(
    #     [[miRNA_list.index(x[0]), drug_list.index(x[1]) + len(miRNA_list)] for x in miRNA_drug.values]).T
    #
    # # 加载 miRNA 的 k-mer 表征向量（每行为一个 miRNA）
    # # 加载 drug 的 GIN 图表示向量（每行为一个 drug）
    # miRNA = pd.read_csv("data/miRNA_kmer.csv", header=None)
    # drug = pd.read_csv("data/drug_GIN_64.csv", header=None)
    # # 将 miRNA 和 drug 特征合并
    # feature = torch.Tensor(miRNA.values.tolist() + drug.values.tolist())

    # miRNA_tensor = torch.tensor(miRNA.values, dtype=torch.float32)
    # drug_tensor = torch.tensor(drug.values, dtype=torch.float32)

    # miRNA_edge_index = fully_connected_edge_index(miRNA_tensor.shape[0])  # [2, num_edges]
    # drug_edge_index = fully_connected_edge_index(drug_tensor.shape[0])  # [2, num_edges]

    # rna3layers = GCN_GAT_GCN(64, 64, 64)
    # drug3layers = GCN_GAT_GCN(64, 64, 64)

    # rna3layers.eval()
    # drug3layers.eval()

    # 2. 不记录计算图（不反向传播）
    # with torch.no_grad():
    # fea_RNA = rna3layers(miRNA_tensor, miRNA_edge_index)
    # fea_drug = drug3layers(drug_tensor, drug_edge_index)

    # feature = torch.cat([fea_RNA, fea_drug], dim=0)

    # num_test=0.2：20% 边用于测试
    # split_labels=True：自动添加正负边标签
    # is_undirected=True：表示图是无向图
    # add_negative_train_samples=False：不添加训练阶段的负采样（在 GAM 中自行构造）
    # train_data, _, test_data = T.RandomLinkSplit(num_val=0, num_test=0.2,
    #                                              is_undirected=True, split_labels=True,
    #                                              add_negative_train_samples=False)(
    #     Data(x=feature, edge_index=edge_index).cuda())
    # .x: 节点特征
    # .edge_index: 正边索引
    # .pos_edge_label_index: 测试正边（用于评估）
    # .neg_edge_label_index: 测试负边（用于评估）
    # splits = dict(train=train_data, test=test_data)
    # return splits


def fully_connected_edge_index(num_nodes):
    edge_index = torch.tensor(list(combinations(range(num_nodes), 2)), dtype=torch.long).T
    # 添加反向边使图无向
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    return edge_index
