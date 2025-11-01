import random

import pandas as pd
import torch.nn as nn
import faiss
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, matthews_corrcoef

from util import *


class NDSGCL(object):
    def __init__(self, conf, training_set, test_set, i):
        # super(NDSGCL, self).__init__(conf, training_set, test_set)
        self.config = conf
        self.emb_size = int(self.config['embbedding.size'])

        #一系列参数
        args = OptionConf(self.config['NCL'])  # NCL=-n_layer 2 -ssl_reg 1e-7 -proto_reg 8e-8 -tau 0.1 -hyper_layers
        # 1 -alpha 1 -num_clusters 5
        self.n_layers = int(args['-n_layer'])  # GCN 层数
        self.ssl_temp = float(args['-tau'])  # 对比学习温度系数 τ
        self.ssl_reg = float(args['-ssl_reg'])  # 对比损失权重
        self.hyper_layers = int(args['-hyper_layers'])  # 用于多层对比学习时取哪个层
        self.alpha = float(args['-alpha'])  # 用于平衡 lncRNA 与 drug 的损失比重
        self.proto_reg = float(args['-proto_reg'])  # 原型对比损失权重
        self.k = int(args['-num_clusters'])  # FAISS 聚类时的簇数量

        self.data = Interaction(conf, training_set, test_set)  # Interaction进行数据处理，构造图，邻接矩阵。util 174行

        # 创建模型，LGCN_Encoder 是轻量图卷积网络
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers)  # NDSGCL 130行
        # 学习率、训练轮数、批大小、L2正则化系数
        self.lRate = float(self.config['learnRate'])
        self.maxEpoch = int(self.config['num.max.epoch'])
        self.batch_size = int(self.config['batch_size'])
        self.reg = float(self.config['reg.lambda'])

        self.lncRNA_centroids = None
        self.lncRNA_2cluster = None
        self.drug_centroids = None
        self.drug_2cluster = None
        self.i = i

    def e_step(self):
        # 获取当前模型中的 lncRNA 和 drug 节点的嵌入向量
        lncRNA_embeddings = self.model.embedding_dict['lncRNA_emb'].detach().cpu().numpy()
        drug_embeddings = self.model.embedding_dict['drug_emb'].detach().cpu().numpy()
        # 对两个嵌入空间分别做 KMeans 聚类
        self.lncRNA_centroids, self.lncRNA_2cluster = self.run_kmeans(lncRNA_embeddings)
        self.drug_centroids, self.drug_2cluster = self.run_kmeans(drug_embeddings)

    # 执行 KMeans 聚类
    def run_kmeans(self, x):
        # 初始化 FAISS 的 KMeans 模型
        # d=self.emb_size：每个向量的维度（嵌入维度）
        # k=self.k：聚类数量
        # gpu=True：启用 GPU 加速（默认使用 device:0）
        kmeans = faiss.Kmeans(d=self.emb_size, k=self.k, gpu=True)
        # 训练 KMeans，确定聚类中心
        kmeans.train(x)
        cluster_cents = kmeans.centroids
        # 利用聚类模型的索引对原始数据x做最近邻搜索
        # I: 每个向量所属的最近聚类中心的编号（聚类标签）
        _, I = kmeans.index.search(x, 1)
        # 将聚类中心和聚类标签转为 CUDA Tensor
        centroids = torch.Tensor(cluster_cents).cuda()
        node2cluster = torch.LongTensor(I).squeeze().cuda()
        return centroids, node2cluster

    def ProtoNCE_loss(self, initial_emb, lncRNA_idx, drug_idx):
        # Embedding 拆分
        lncRNA_emb, drug_emb = torch.split(initial_emb, [self.data.lncRNA_num, self.data.drug_num])
        # 获取 lncRNA 和药物的聚类中心
        # self.lncRNA_2cluster[lncRNA_idx] 获取每个 lncRNA 节点所属的聚类索引。
        # self.lncRNA_centroids[lncRNA2cluster] 获取这些 lncRNA 节点对应的聚类中心。
        lncRNA2cluster = self.lncRNA_2cluster[lncRNA_idx]
        lncRNA2centroids = self.lncRNA_centroids[lncRNA2cluster]
        # 计算 InfoNCE 损失
        proto_nce_loss_lncRNA = InfoNCE(lncRNA_emb[lncRNA_idx],lncRNA2centroids,self.ssl_temp)  # InfoNCE util 124行
        # drug 同理
        drug2cluster = self.drug_2cluster[drug_idx]
        drug2centroids = self.drug_centroids[drug2cluster]
        proto_nce_loss_drug = InfoNCE(drug_emb[drug_idx],drug2centroids,self.ssl_temp)
        proto_nce_loss = self.proto_reg * (proto_nce_loss_lncRNA + proto_nce_loss_drug)
        return proto_nce_loss

    def ssl_layer_loss(self, context_emb, initial_emb, lncRNA, drug):
        # context_emb 多层GCN后的嵌入，带有图结构上下文信息
        # initial_emb 初始嵌入，仅从输入 embedding 初始化，尚未图传播
        # lncRNA, drug 一个 batch 中正样本对的索引列表

        # 切分嵌入表示
        context_lncRNA_emb_all, context_drug_emb_all = torch.split(context_emb, [self.data.lncRNA_num, self.data.drug_num])
        initial_lncRNA_emb_all, initial_drug_emb_all = torch.split(initial_emb, [self.data.lncRNA_num, self.data.drug_num])
        # 构造正负样本
        context_lncRNA_emb = context_lncRNA_emb_all[lncRNA]
        initial_lncRNA_emb = initial_lncRNA_emb_all[lncRNA]
        # 对嵌入进行 L2 归一化
        norm_lncRNA_emb1 = F.normalize(context_lncRNA_emb)
        norm_lncRNA_emb2 = F.normalize(initial_lncRNA_emb)
        norm_all_lncRNA_emb = F.normalize(initial_lncRNA_emb_all)

        # 正样本对
        pos_score_lncRNA = torch.mul(norm_lncRNA_emb1, norm_lncRNA_emb2).sum(dim=1)

        # 将当前 batch 中的表示与所有 lncRNA 节点的表示进行点积，形成对比矩阵
        ttl_score_lncRNA = torch.matmul(norm_lncRNA_emb1, norm_all_lncRNA_emb.transpose(0, 1))

        # 温度缩放 + Softmax
        pos_score_lncRNA = torch.exp(pos_score_lncRNA / self.ssl_temp)
        ttl_score_lncRNA = torch.exp(ttl_score_lncRNA / self.ssl_temp).sum(dim=1)

        # 最终损失
        ssl_loss_lncRNA = -torch.log(pos_score_lncRNA / ttl_score_lncRNA).sum()

        # drug 同理
        context_drug_emb = context_drug_emb_all[drug]
        initial_drug_emb = initial_drug_emb_all[drug]
        norm_drug_emb1 = F.normalize(context_drug_emb)
        norm_drug_emb2 = F.normalize(initial_drug_emb)
        norm_all_drug_emb = F.normalize(initial_drug_emb_all)
        pos_score_drug = torch.mul(norm_drug_emb1, norm_drug_emb2).sum(dim=1)
        ttl_score_drug = torch.matmul(norm_drug_emb1, norm_all_drug_emb.transpose(0, 1))
        pos_score_drug = torch.exp(pos_score_drug / self.ssl_temp)
        ttl_score_drug = torch.exp(ttl_score_drug / self.ssl_temp).sum(dim=1)
        ssl_loss_drug = -torch.log(pos_score_drug / ttl_score_drug).sum()

        # 最终对比损失
        ssl_loss = self.ssl_reg * (ssl_loss_lncRNA + self.alpha * ssl_loss_drug)
        return ssl_loss

    def train(self):
        model = self.model.cuda()  # NDSGCL 26行
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)  # 优化器
        for epoch in range(self.maxEpoch):
            self.e_step()  # NDSGCL 39行

            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):  # next_batch_pairwise util 70行
                lncRNA_idx, pos_idx, neg_idx = batch
                model.train()
                # rec_lncRNA_emb, rec_drug_emb：最终嵌入 emb_list: 多层 LGCN 中间层结果
                rec_lncRNA_emb, rec_drug_emb, emb_list = model()
                lncRNA_emb, pos_drug_emb, neg_drug_emb = rec_lncRNA_emb[lncRNA_idx], rec_drug_emb[pos_idx], rec_drug_emb[neg_idx]
                # BPR Loss
                rec_loss = bpr_loss(lncRNA_emb, pos_drug_emb, neg_drug_emb)  # util 104行
                # 对比学习损失（SSL）
                """Contrastive Learning With Local Structural Neighbours"""
                initial_emb = emb_list[0]
                context_emb = emb_list[self.hyper_layers*2]
                ssl_loss = self.ssl_layer_loss(context_emb,initial_emb,lncRNA_idx,pos_idx)
                # 原型对比损失（ProtoNCE）
                proto_loss = self.ProtoNCE_loss(initial_emb, lncRNA_idx, pos_idx)
                # 计算总损失
                batch_loss = rec_loss + l2_reg_loss(self.reg, lncRNA_emb, pos_drug_emb) + ssl_loss + proto_loss
                # 反向传播与参数更新
                optimizer.zero_grad()  # 清除上一个 batch 的梯度
                batch_loss.backward()  # 反向传播，计算梯度
                optimizer.step()  # 根据梯度更新模型参数
                if n % 100 == 0:
                    print('training:', epoch + 1, 'batch', n,
                          'rec_loss:', rec_loss.item(),
                          'ssl_loss', ssl_loss.item(),
                          'proto_loss', proto_loss.item())

            # 保存最终模型嵌入
            model.eval()
            with torch.no_grad():
                self.lncRNA_emb, self.drug_emb, _ = model()
            # print(self.lncRNA_emb.shape)
            # print(self.drug_emb.shape)
            auc, aupr, acc, sen, pre, spe, F1, mcc = test(self.data, self.lncRNA_emb, self.drug_emb)
            print("AUC:", round(float(auc), 4))
            print("AUPR:", round(float(aupr), 4))
            print("ACC:", round(float(acc), 4))
            print("Sensitivity (Recall):", round(float(sen), 4))
            print("Precision:", round(float(pre), 4))
            print("Specificity:", round(float(spe), 4))
            print("F1-score:", round(float(F1), 4))
            print("MCC:", round(float(mcc), 4))


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

@torch.no_grad()
def test(data, lncRNA_emb, drug_emb, score_type='dot', neg_ratio=1):
    data = torch.load("MiDrug_test_data.pth", weights_only=False)
    edge_label = data[("miRNA", "MiDrug", "drug")].edge_label
    edge_label_index = data[("miRNA", "MiDrug", "drug")].edge_label_index
    pos_pred = []
    neg_pred = []
    pos_rows = []
    neg_rows = []
    # 正样本：从 edge_label_index 中提取标签为 1 的边
    pos_mask = edge_label == 1
    pos_edges = edge_label_index[:, pos_mask].cpu().numpy()
    num_pos = pos_edges.shape[1]

    # 正样本处理
    for lnc_idx, drug_idx in pos_edges.T:  # .T 转置，便于逐个处理
        # if lnc_idx >= lncRNA_emb.shape[0] or drug_idx >= drug_emb.shape[0]:
        #     continue
        lnc_vec = lncRNA_emb[lnc_idx]
        drug_vec = drug_emb[drug_idx]
        score = torch.mul(lnc_vec, drug_vec).sum() if score_type == 'dot' else F.cosine_similarity(
            lnc_vec.unsqueeze(0), drug_vec.unsqueeze(0)).item()
        pos_pred.append(score)
        pos_rows.append((lnc_idx, drug_idx, score, 1))

    # 负样本：从 edge_label_index 中提取标签为 0 的边
    neg_mask = edge_label == 0
    neg_edges = edge_label_index[:, neg_mask].cpu().numpy()

    # 负样本处理
    for lnc_idx, drug_idx in neg_edges.T:  # .T 转置，便于逐个处理
        # if lnc_idx >= lncRNA_emb.shape[0] or drug_idx >= drug_emb.shape[0]:
        #     continue
        lnc_vec = lncRNA_emb[lnc_idx]
        drug_vec = drug_emb[drug_idx]
        score = torch.mul(lnc_vec, drug_vec).sum() if score_type == 'dot' else F.cosine_similarity(
            lnc_vec.unsqueeze(0), drug_vec.unsqueeze(0)).item()
        neg_pred.append(score)
        neg_rows.append((lnc_idx, drug_idx, score, 0))

    # 转换成 tensor
    pos_pred = torch.tensor(pos_pred)
    neg_pred = torch.tensor(neg_pred)

    # 合并正负样本预测分数
    pred = torch.cat([pos_pred, neg_pred], dim=0)

    # 标准化、二值化
    pred = torch.sigmoid(pred.clone().detach())

    # 创建真实标签：正样本标签为 1，负样本标签为 0
    y = torch.cat([pos_pred.new_ones(pos_pred.size(0)), neg_pred.new_zeros(neg_pred.size(0))], dim=0)

    # 转换为 numpy 格式，计算 AUC、AUPR 等
    y, pred = y.cpu().numpy(), pred.cpu().numpy()
    # 合并预测结果
    all_rows = pos_rows + neg_rows
    df = pd.DataFrame(all_rows, columns=['lncRNA_id', 'drug_id', 'score', 'label'])

    tensor_list = df['score'].tolist()  # 每个元素是 tensor(..., device='cuda:0')

    # 步骤2：堆叠张量并移到CPU（自动处理CUDA设备）
    stacked_tensors = torch.stack(tensor_list).cpu()  # shape: [n_samples]
    normalized_scores = (stacked_tensors - stacked_tensors.min()) / (stacked_tensors.max() - stacked_tensors.min())
    # 对预测分数应用 sigmoid（变为概率）
    df['score'] = normalized_scores

    # 保存到 CSV
    df.to_csv('lncRNA_results.csv', index=False)

    auc = roc_auc_score(y, pred)
    aupr = average_precision_score(y, pred)
    temp = torch.tensor(pred)
    temp[temp >= 0.5] = 1
    temp[temp < 0.5] = 0
    acc, sen, pre, spe, F1, mcc = calculate_metrics(y, temp.cpu())
    return auc, aupr, acc, sen, pre, spe, F1, mcc


class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = convert_sparse_mat_to_tensor(self.norm_adj).cuda()  # util 132行

    # 初始化lncRNA和drug的嵌入向量
    def _init_model(self):
        initializer = nn.init.xavier_uniform_  # 使用 Xavier uniform 初始化方法对参数进行初始化
        # 创建并初始化两个嵌入矩阵（embedding matrix）：
        # lncRNA_emb：所有 lncRNA 的表示向量，形状为 [lncRNA_num, latent_size]
        # drug_emb：所有 drug 的表示向量，形状为 [drug_num, latent_size]

        # nn.ParameterDict(...)：将lncRNA_emb和drug_emb封装在一个参数字典中
        embedding_dict = nn.ParameterDict({
            # nn.Parameter(...)：将其注册为模型的参数（model.parameters() 中可被优化器自动更新）
            'lncRNA_emb': nn.Parameter(initializer(torch.empty(self.data.lncRNA_num, self.latent_size))),
            'drug_emb': nn.Parameter(initializer(torch.empty(self.data.drug_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        # 初始嵌入拼接在一起形成整体图中所有节点的 embedding
        ego_embeddings = torch.cat([self.embedding_dict['lncRNA_emb'], self.embedding_dict['drug_emb']], 0)
        # 保存第 0 层（原始）嵌入
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            # sparse.mm 表示稀疏矩阵乘法，计算 A * X
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        # 将所有层的 embedding 沿新维度叠加
        lgcn_all_embeddings = torch.stack(all_embeddings, dim=1)
        # 对所有层的表示做平均，融合多层特征
        lgcn_all_embeddings = torch.mean(lgcn_all_embeddings, dim=1)
        # 将最终融合的嵌入分为两部分：lncRNA 的嵌入 drug 的嵌入
        lncRNA_all_embeddings = lgcn_all_embeddings[:self.data.lncRNA_num]
        drug_all_embeddings = lgcn_all_embeddings[self.data.lncRNA_num:]
        return lncRNA_all_embeddings, drug_all_embeddings, all_embeddings