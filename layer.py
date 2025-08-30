import torch
import dgl.nn as dglnn
from torch import nn
from torch.testing._internal.common_subclass import SparseTensor
import torch
from torch_geometric.nn import Linear, GCNConv, SAGEConv, GATConv, GINConv, GATv2Conv
import torch.nn.functional as F


class HeteroLinear(nn.Module):
    """Apply linear transformations on heterogeneous inputs.
    """

    def __init__(self, in_feats, hidden_feats, dropout=0., bn=False):
        """
        Parameters
        ----------
        in_feats : dict[key, int]
            Input feature size for heterogeneous inputs. A key can be a string or a tuple of strings.
        hidden_feats : int
            Output feature size.
        dropout : int
            The dropout rate.
        bn : bool
            Use batch normalization or not.
        """
        super(HeteroLinear, self).__init__()
        self.linears = nn.ModuleDict()
        # 为每种节点类型创建独立的线性层
        for typ, typ_in_size in in_feats.items():
            self.linears[str(typ)] = nn.Linear(typ_in_size, hidden_feats)
            nn.init.xavier_uniform_(self.linears[str(typ)].weight)

        # 可选的批归一化和Dropout
        if bn:
            self.bn = nn.BatchNorm1d(hidden_feats)
        else:
            self.bn = False
        if dropout != 0.:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = False

    def forward(self, feat):
        out_feat = dict()
        for typ, typ_feat in feat.items():
            out_feat[typ] = self.linears[str(typ)](typ_feat)
            if self.bn:
                out_feat[typ] = self.bn(out_feat[typ])
            if self.dropout:
                out_feat[typ] = self.dropout(out_feat[typ])
        return out_feat


class Node_Embedding(nn.Module):
    """HeteroGCN block.
    """

    def __init__(self, rel_names, in_feats, hidden_feats, dropout=0., bn=False):
        """
        Parameters
        ----------
        in_feats : int
            Input feature size.
        hidden_feats : int
            Output feature size.
        dropout : int
            The dropout rate.
        bn : bool
            Use batch normalization or not.
        """
        super().__init__()
        HeteroGraphdict = {}
        for rel in rel_names:
            graphconv = dglnn.GraphConv(in_feats, hidden_feats)
            nn.init.xavier_normal_(graphconv.weight)
            HeteroGraphdict[rel] = graphconv
        self.embedding = dglnn.HeteroGraphConv(HeteroGraphdict, aggregate='sum')
        self.prelu = nn.PReLU()
        if bn:
            self.bn = nn.BatchNorm1d(hidden_feats)
        else:
            self.bn = False
        if dropout != 0.:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = False

    def forward(self, graph, inputs):
        h = self.embedding(graph, inputs)
        if self.bn:
            h = {k: self.bn(v) for k, v in h.items()}
        if self.dropout:
            h = {k: self.dropout(v) for k, v in h.items()}
        h = {k: self.prelu(v) for k, v in h.items()}
        return h


class LayerAttention(nn.Module):
    """Layer attention block.
    """

    def __init__(self, in_feats, hidden_feats=128):
        """
        Parameters
        ----------
        in_feats : int
            Input feature size.
        hidden_feats : int
            Output feature size.
        """
        super(LayerAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_feats, hidden_feats),
            nn.Tanh(),
            nn.Linear(hidden_feats, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)


class MetaPathAggregator(nn.Module):
    """Aggregating the meta-path instances in each bags.
    """

    def __init__(self, in_feats, hidden_feats, agg_type='sum', dropout=0., bn=False):
        """
        Parameters
        ----------
        in_feats : int
            Input feature size.
        hidden_feats : int
            Output feature size.
        agg_type : ["sum", "mean", "Linear", "BiTrans"]
            The aggregator to be used.
        dropout : int
            The dropout rate.
        bn : bool
            Use batch normalization or not.
        """
        super(MetaPathAggregator, self).__init__()
        self.agg_type = agg_type
        if agg_type == 'sum':
            self.aggregator = torch.sum
        elif agg_type == 'mean':
            self.aggregator = torch.mean
        elif agg_type == 'Linear':
            self.aggregator = nn.Linear(in_feats * 4, hidden_feats, bias=False)
            nn.init.xavier_uniform_(self.aggregator.weight)
        elif agg_type == 'BiTrans':
            self.aggregator_drug_disease = nn.Linear(in_feats, hidden_feats, bias=False)
            self.aggregator_disease_drug = nn.Linear(in_feats, hidden_feats, bias=False)
            self.aggregator_drug = nn.Linear(in_feats, int(hidden_feats / 2), bias=False)
            self.aggregator_dis = nn.Linear(in_feats, int(hidden_feats / 2), bias=False)
            nn.init.xavier_uniform_(self.aggregator_drug_disease.weight)
            nn.init.xavier_uniform_(self.aggregator_disease_drug.weight)
            nn.init.xavier_uniform_(self.aggregator_drug.weight)
            nn.init.xavier_uniform_(self.aggregator_dis.weight)

    def forward(self, feature, mp_ins):
        mp_ins_miRNA = mp_ins[:, :, :1]
        mp_ins_gene1 = mp_ins[:, :, 1:2]
        mp_ins_gene2 = mp_ins[:, :, 2:3]
        mp_ins_drug = mp_ins[:, :, 3:]
        mp_ins_feat = torch.cat([feature['miRNA'][mp_ins_miRNA],
                                 feature['gene'][mp_ins_gene1],
                                 feature['gene'][mp_ins_gene2],
                                 feature['drug'][mp_ins_drug]], dim=2)
        if self.agg_type in ['sum', 'mean']:
            ins_emb = self.aggregator(mp_ins_feat, dim=2)
        elif self.agg_type == 'Linear':
            ins_emb = self.aggregator(mp_ins_feat.reshape(mp_ins_feat.shape[0], mp_ins_feat.shape[1],
                                                          mp_ins_feat.shape[2] * mp_ins_feat.shape[3]))
        else:
            hd_feat = mp_ins_feat.shape[3]
            mp_ins_feat = mp_ins_feat.reshape(mp_ins_feat.shape[0], mp_ins_feat.shape[1],
                                              mp_ins_feat.shape[2] * mp_ins_feat.shape[3])

            dis_feat = (((self.aggregator_drug_disease((mp_ins_feat[:, :, :hd_feat] +
                                                        mp_ins_feat[:, :, hd_feat:hd_feat * 2]) / 2)
                          + mp_ins_feat[:, :, hd_feat * 2:hd_feat * 3]) / 2)
                        + mp_ins_feat[:, :, hd_feat * 3:]) / 2
            drug_feat = (((self.aggregator_disease_drug((mp_ins_feat[:, :, hd_feat * 3:]
                                                         + mp_ins_feat[:, :, hd_feat * 2:hd_feat * 3]) / 2)
                           + mp_ins_feat[:, :, hd_feat:hd_feat * 2]) / 2)
                         + mp_ins_feat[:, :, :hd_feat]) / 2


            ins_emb = torch.cat((self.aggregator_drug(drug_feat),
                                 self.aggregator_dis(dis_feat)), dim=2)
        return ins_emb


class MILNet(nn.Module):
    """Attention based instance aggregation block for bag embedding.
    """

    def __init__(self, in_feats, hidden_feats):
        """
        Parameters
        ----------
        in_feats : int
            Input feature size.
        hidden_feats : int
            Output feature size.
        """
        super(MILNet, self).__init__()

        self.project = nn.Sequential(nn.Linear(in_feats, hidden_feats),
                                     nn.Tanh(),
                                     nn.Linear(hidden_feats, 1, bias=False))

    def forward(self, ins_emb, output_attn=True):
        attn = torch.softmax(self.project(ins_emb), dim=1)
        bag_emb = (ins_emb * attn).sum(dim=1)
        if output_attn:
            return bag_emb, attn
        else:
            return bag_emb


class InstanceNet(nn.Module):
    """Instance predictor.
    """

    def __init__(self, in_feats, k=3):
        """
        Parameters
        ----------
        in_feats : int
            Input feature size.
        k : int
            A topk filtering used in the aggregation of predictions.
        """
        super(InstanceNet, self).__init__()

        self.k = k
        self.weights = nn.Linear(int(in_feats / 2), int(in_feats / 2), bias=False)
        nn.init.xavier_uniform_(self.weights.weight)

    def forward(self, ins_emb, attn):
        miRNA_ins = ins_emb[:, :, :int(ins_emb.shape[-1] / 2)]
        drug_ins = ins_emb[:, :, int(ins_emb.shape[-1] / 2):]
        pred = torch.matmul(self.weights(miRNA_ins).reshape(miRNA_ins.shape[0], miRNA_ins.shape[1],
                                                            1, miRNA_ins.shape[2]),
                            drug_ins.unsqueeze(dim=-1)).squeeze(dim=3)
        attn_pred = attn * pred
        topk_out = torch.mean(attn_pred.topk(k=self.k, dim=1)[0], dim=1)
        return topk_out


class MLP(nn.Module):
    """Bag predictor.
    """

    def __init__(self, in_feats, dropout=0.):
        """
        Parameters
        ----------
        in_feats : int
            Input feature size.
        dropout : int
            The dropout rate.
        """
        super(MLP, self).__init__()
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = False
        self.linear = nn.Linear(in_feats, 1, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, bag_emb):
        if self.dropout:
            bag_emb = self.dropout(bag_emb)
        outputs = self.linear(bag_emb)
        return outputs


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

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, layer):
        super(GNNEncoder, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            heads = 1 if i == num_layers - 1 or 'gat' not in layer else 4

            self.convs.append(creat_gnn_layer(layer, first_channels, second_channels, heads))
            self.bns.append(torch.nn.BatchNorm1d(second_channels * heads))
        self.dropout = torch.nn.Dropout(0.5)
        self.activation = torch.nn.ELU()

    def forward(self, x, edge_index):
        # edge_index = SparseTensor.from_edge_index(edge_index, sparse_sizes=(x.size(0), x.size(0))).cuda()
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

        x = z[edge[0]] * z[edge[1]]
        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)
        x = self.mlps[-1](x)
        return x
