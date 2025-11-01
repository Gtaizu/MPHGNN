import pandas as pd
import torch
import torch.nn.functional as F
from utils import calculate_metrics, draw_auc, draw_aupr
from torch_sparse import SparseTensor
from torch.utils.data import DataLoader
from torch_geometric.nn import Linear, GCNConv, SAGEConv, GATConv, GINConv, GATv2Conv
from torch_geometric.utils import add_self_loops, negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score


def creat_gnn_layer(name, first_channels, second_channels, heads):
    if name == "sage":
        layer = SAGEConv(first_channels, second_channels)
    elif name == "gcn":
        layer = GCNConv(first_channels, second_channels)
    elif name == "gin":
        layer = GINConv(Linear(first_channels, second_channels), train_eps=True)
    elif name == "gat":
        layer = GATConv(-1, second_channels, heads=heads)
    elif name == "gat2":
        layer = GATv2Conv(-1, second_channels, heads=heads)
    else:
        raise ValueError(name)
    return layer


class GNNEncoder(torch.nn.Module):
    # in_channels: 输入特征维度
    # hidden_channels: 中间隐藏层维度
    # out_channels: 最后一层输出维度
    # num_layers: GNN 层数
    # layer: GNN 类型名称（如 "gcn"、"gat" 等）
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, layer):
        super(GNNEncoder, self).__init__()
        # 分别存储 GNN 层 和 BatchNorm 层
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        # 动态构建每一层
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            heads = 1 if i == num_layers - 1 or 'gat' not in layer else 4

            self.convs.append(creat_gnn_layer(layer, first_channels, second_channels, heads))
            self.bns.append(torch.nn.BatchNorm1d(second_channels * heads))
        # Dropout + 激活函数
        self.dropout = torch.nn.Dropout(0.5)
        self.activation = torch.nn.ELU()

    def forward(self, x, edge_index):
        # 将 PyG 的 edge_index 转为 SparseTensor
        edge_index = SparseTensor.from_edge_index(edge_index, sparse_sizes=(x.size(0), x.size(0))).cuda()
        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
        x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = self.activation(x)
        return x

# 边解码器
class EdgeDecoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(EdgeDecoder, self).__init__()
        self.mlps = torch.nn.ModuleList()

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.mlps.append(torch.nn.Linear(first_channels, second_channels))

        self.dropout = torch.nn.Dropout(0.5)
        self.activation = torch.nn.ELU()

    def forward(self, z, edge):
        # z：所有节点的嵌入向量，shape 为 [num_nodes, embed_dim]。
        # edge：边的索引，shape 为 [2, num_edges]，表示每条边的两个节点索引

        # z[edge[0]]：取出每条边的起点节点的嵌入，shape 为 [num_edges, embed_dim]。
        # z[edge[1]]：终点节点的嵌入
        x = z[edge[0]] * z[edge[1]]
        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)
        x = self.mlps[-1](x)
        return x

# 二元交叉熵损失函数
def ce_loss(pos_out, neg_out):
    # pos_out.sigmoid()：将模型输出的实数值映射到 [0, 1] 范围，表示正边存在的概率。
    # torch.ones_like(pos_out)：构造一个和 pos_out 同 shape 的全 1 tensor，作为标签（ground truth）。
    # F.binary_cross_entropy(...)：对每个正边计算 BCE(p, 1)。
    pos_loss = F.binary_cross_entropy(pos_out.sigmoid(), torch.ones_like(pos_out))
    neg_loss = F.binary_cross_entropy(neg_out.sigmoid(), torch.zeros_like(neg_out))
    return pos_loss + neg_loss


class GAM(torch.nn.Module):
    def __init__(self, encoder, edge_decoder, mask):
        super(GAM, self).__init__()
        self.encoder = encoder
        self.edge_decoder = edge_decoder
        self.mask = mask
        self.loss_fn = ce_loss
        self.negative_sampler = negative_sampling

    def forward(self, x, edge_index):
        return self.encoder(x, edge_index)

    # 一轮训练逻辑
    def train_epoch(self, data, optimizer, batch_size=2 ** 16, grad_norm=1.0):
        x, edge_index = data['train'].x, data['train'].edge_index
        # 边遮蔽
        remaining_edges, masked_edges = self.mask(edge_index)
        all_existing = torch.cat([data['train'].edge_index, data['test'].pos_edge_label_index, data['test'].neg_edge_label_index], dim=1)
        loss_total = 0.0
        # 负采样
        # 增加自环
        aug_edge_index, _ = add_self_loops(all_existing)
        # 生成和被 mask 掉的正样本数量一样多的负样本
        neg_edges = self.negative_sampler(
            aug_edge_index, num_nodes=data['train'].num_nodes, num_neg_samples=masked_edges.view(2, -1).size(1)
        ).view_as(masked_edges)

        for perm in DataLoader(range(masked_edges.size(1)), batch_size=batch_size, shuffle=True):
            optimizer.zero_grad()
            z = self.encoder(x, remaining_edges)  # 编码器输入只使用未遮蔽的图结构
            batch_masked_edges = masked_edges[:, perm]
            batch_neg_edges = neg_edges[:, perm]

            pos_out = self.edge_decoder(z, batch_masked_edges)  # 正样本预测
            neg_out = self.edge_decoder(z, batch_neg_edges)  # 负样本预测
            # 计算BCE损失、反向传播、梯度裁剪（防止梯度爆炸）、优化器更新
            loss = self.loss_fn(pos_out, neg_out)
            loss.backward()
            # self.parameters()：模型中所有需要训练的参数。
            # grad_norm：最大允许的梯度范数（比如 1.0），是设定的阈值。
            # clip_grad_norm_()：这个函数会：计算所有参数的梯度总范数（L2 范数），如果这个范数大于你设置的阈值 grad_norm
            # 就按比例缩小所有梯度，使总范数 <= 这个阈值
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_norm)  # 梯度裁剪
            optimizer.step()
            loss_total += loss.item()
        return loss_total

    @torch.no_grad()
    # 批量计算节点对（边）的存在概率预测结果
    # z：所有节点的嵌入表示（由encoder输出），shape为[num_nodes, embed_dim]。
    # edges：待预测的边，shape为[2, num_edges]，每列是一个边的起点终点(i, j)。
    def batch_predict(self, z, edges, batch_size=2 ** 16):
        preds = []
        for perm in DataLoader(range(edges.size(1)), batch_size):
            edge = edges[:, perm]
            preds += [self.edge_decoder(z, edge).squeeze().cpu()]
        pred = torch.cat(preds, dim=0)
        return pred

    @torch.no_grad()
    # z：所有节点的 embedding（来自 encoder）。
    # pos_edge_index：测试集中真实存在的边（正样本）。
    # neg_edge_index：负采样得到的、实际上不存在的边（负样本）。
    # l：图神经网络类型的标签，用于绘图时显示。
    def test(self, z, pos_edge_index, neg_edge_index, l='gcn'):
        pos_pred = self.batch_predict(z, pos_edge_index)
        neg_pred = self.batch_predict(z, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y = torch.cat([pos_pred.new_ones(pos_pred.size(0)), neg_pred.new_zeros(neg_pred.size(0))], dim=0)

        y, pred = y.cpu().numpy(), pred.cpu().numpy()
        draw_auc(y, pred, l)
        draw_aupr(y, pred, l)
        auc = roc_auc_score(y, pred)
        aupr = average_precision_score(y, pred)
        temp = torch.tensor(pred)
        temp[temp >= 0.5] = 1
        temp[temp < 0.5] = 0
        acc, sen, pre, spe, F1, mcc = calculate_metrics(y, temp.cpu())

        # 合并正负边索引
        all_edges = torch.cat([pos_edge_index, neg_edge_index], dim=1)  # shape: [2, num_edges]
        all_labels = torch.cat([torch.ones(pos_edge_index.size(1)), torch.zeros(neg_edge_index.size(1))], dim=0)

        # 转为 numpy
        data = torch.load("data/MiDrug_test_data.pth", weights_only=False)
        num_miRNA = data['miRNA'].num_nodes
        src_nodes = all_edges[0].cpu().numpy()
        dst_nodes = all_edges[1].cpu().numpy()
        dst_nodes = dst_nodes - num_miRNA
        scores = pred
        scores_normalized = (pred - pred.min()) / (pred.max() - pred.min())
        scores = scores_normalized
        labels = all_labels.cpu().numpy()

        # 构建 DataFrame
        df = pd.DataFrame({
            'miRNA_id': src_nodes,
            'drug_id': dst_nodes,
            'score': scores,
            'label': labels
        })

        # 保存到文件
        df.to_csv('miRNA_drug_prediction_results.csv', index=False)

        return [auc, aupr, acc.item(), sen.item(), pre.item(), spe.item(), F1.item(), mcc.item()]

