import numpy as np
import scipy.sparse as sp
import torch
import math
import argparse
from sklearn.metrics import precision_recall_curve, roc_curve, accuracy_score, f1_score, auc, precision_score, \
    recall_score, matthews_corrcoef, confusion_matrix


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run GAT-LGA.")
    parser.add_argument("--dataset_path",nargs="?",default="../Datasets/Dataset2/",help="Training datasets.")#Dataset1,Dataset2
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

# 将一个相似度矩阵转换为图神经网络 (GNN) 所需的边索引 (edge_index) 格式
def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # 如果相似度大于 0.5，就认为节点 i 和 j 之间存在一条边
            if matrix[i][j] > 0.5:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.tensor(np.array(edge_index))

# 将一个相似度矩阵转化为一个二值邻接矩阵
def get_adj(matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] > 0.5:
                matrix[i][j] = 1
            else:
                matrix[i][j] = 0
    return torch.tensor(matrix)

# 高斯核函数
def gaussian_sim(data_list):
    # data_list：一个二值的 lncRNA-drug 关联矩阵，形状为 (lnc数量, drug数量)
    '''
    calculate the gaussian similarity
    '''
    print("Similarity calculation!")


    nl = data_list.shape[0]  # lncRNA数量
    nd = data_list.shape[1]  # drug数量
    sl = [0] * nl  # 每个lnc的平方范数
    sd = [0] * nd  # 每个drug的平方范数
    pkl = np.zeros((nl, nl))  # lncRNA相似度矩阵初始化
    pkd = np.zeros((nd, nd))  # drug相似度矩阵初始化
    # 计算 lncRNA 间相似度
    X = np.array(data_list)
    Y = np.array(data_list)
    D = np.sum(X*X, axis=1, keepdims=True) \
            + np.sum(Y*Y, axis=1, keepdims=True).T \
            - 2 * np.dot(X, Y.T)
    for i in range(nl):
        sl[i]=pow(np.linalg.norm(data_list[i,:]),2)
    gamal=sum(sl)/nl
    pkl=np.exp(-gamal*D)
    
    X = np.array(data_list.T)
    Y = np.array(data_list.T)
    D2 = np.sum(X*X, axis=1, keepdims=True) \
            + np.sum(Y*Y, axis=1, keepdims=True).T \
            - 2 * np.dot(X, Y.T)
    for i in range(nd):
        sd[i]=pow(np.linalg.norm(data_list[:,i]),2)
    gamad=sum(sd)/nd 
    pkd=np.exp(-gamad*D2)
    print("Finish similarity calculation!")
    #np.savetxt('../Datasets/Dataset1/lnc_feat.txt',pkl,fmt="%.18f",delimiter="\t")
    #np.savetxt('../Datasets/Dataset1/drug_feat.txt',pkd,fmt="%.18f",delimiter="\t")
    return pkl, pkd


def get_negative_samples(adj_matrix, num_samples):
    # 获取所有 adj_matrix == 0 的索引（即不存在的边）
    zero_indices = (adj_matrix == 0).nonzero(as_tuple=False)  # 形状 [num_zeros, 2]

    # 随机选择 num_samples 个负样本
    rand_indices = torch.randperm(zero_indices.size(0))[:num_samples]
    neg_edge_index = zero_indices[rand_indices].t()  # 转为 [2, num_samples]

    return neg_edge_index


def load_data(args):
    print('Loading {}...'.format(args.dataset_path))

    # # 加载 lncRNA-drug 的关联矩阵，格式为 CSV
    # LG_matrix = np.loadtxt(open(args.dataset_path + 'lnc_drug_net.csv',"rb"),delimiter=",",skiprows=0)
    # # 随机打乱 lncRNA-drug 的顺序
    # reorder = np.arange(LG_matrix.shape[0])
    # np.random.shuffle(reorder)
    # LG_matrix_all = LG_matrix[reorder,:]
    
    #labels=LG_matrix    
    #l_feat=np.hstack((l_m_feat,l_s_feat))
    #g_feat=np.hstack((g_m_feat,g_s_feat))
    #F1=l_s_feat
    #F2=g_s_feat

    data = torch.load("ncRNADrug2.pth", weights_only=False)
    edge_index = data[("ncRNA", "ncDrug", "drug")].edge_index
    num_edges = edge_index.size(1)

    # 随机打乱边的索引
    shuffled_indices = torch.randperm(num_edges)
    # 按比例划分（例如 80% 训练，20% 验证）
    train_ratio = 0.8
    train_size = int(num_edges * train_ratio)

    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:]

    # 获取训练集和验证集的 edge_index
    train_edge_index = edge_index[:, train_indices]
    val_edge_index = edge_index[:, val_indices]
    val_ratio = 0.2  # 验证集比例
    val_size = int(num_edges * val_ratio)

    # 训练集和验证集的正边
    train_edge_index_pos = edge_index[:, train_indices]
    val_edge_index_pos = edge_index[:, val_indices]

    # 提取 edge_label_index 和 edge_label
    edge_label_index = data[("ncRNA", "ncDrug", "drug")].edge_label_index
    edge_label = data[("ncRNA", "ncDrug", "drug")].edge_label

    # 找到标签为 1 的边的索引位置
    pos_indices = (edge_label == 1).nonzero(as_tuple=True)[0]
    neg_indices = (edge_label == 0).nonzero(as_tuple=True)[0]
    # 提取对应的边（正样本边）
    pos_edge_index = edge_label_index[:, pos_indices]
    neg_edge_index = edge_label_index[:, neg_indices]
    all_index = torch.cat([edge_index, pos_edge_index], dim=1)

    # # 获取所有涉及的 lncRNA 和 drug 节点
    # lncRNA_nodes = edge_index[0].unique()
    # drug_nodes = edge_index[1].unique()
    #
    # # 重新编号映射：旧索引 -> 新索引（0~N-1）
    # lncRNA_map = {old.item(): new for new, old in enumerate(lncRNA_nodes)}
    # drug_map = {old.item(): new for new, old in enumerate(drug_nodes)}
    #
    # # 应用映射：构建新索引的 edge 列表
    # new_lncRNA_idx = torch.tensor([lncRNA_map[i.item()] for i in edge_index[0]])
    # new_drug_idx = torch.tensor([drug_map[i.item()] for i in edge_index[1]])
    #
    # # 构建稠密关联矩阵
    # adj_matrix = torch.zeros((len(lncRNA_nodes), len(drug_nodes)), dtype=torch.float32)
    # adj_matrix[new_lncRNA_idx, new_drug_idx] = 1.0


    # 获取 lncRNA 和 drug 的节点数
    num_lnc = data["ncRNA"].num_nodes
    num_drug = data["drug"].num_nodes
    # 初始化邻接矩阵，shape = [num_lnc, num_drug]
    adj_matrix = torch.zeros((num_lnc, num_drug), dtype=torch.float32)
    adj_matrix_test = torch.zeros((num_lnc, num_drug), dtype=torch.float32)
    all_matrix = torch.zeros((num_lnc, num_drug), dtype=torch.float32)

    # edge_index[0] 是 lncRNA 索引，edge_index[1] 是 drug 索引
    adj_matrix[edge_index[0], edge_index[1]] = 1.0
    adj_matrix_test[pos_edge_index[0], pos_edge_index[1]] = 1.0
    all_matrix[all_index[0], all_index[1]] = 1.0
    all_matrix[neg_edge_index[0], neg_edge_index[1]] = 0.0
    val_neg_edge_index = get_negative_samples(adj_matrix, val_size)
    val_edge_index = torch.cat([val_edge_index_pos, val_neg_edge_index], dim=1)
    LG_matrix = adj_matrix.cpu().numpy()
    LG_matrix_test = adj_matrix_test.cpu().numpy()
    LG_matrix_all = all_matrix.cpu().numpy()


    # reorder = np.arange(LG_matrix_all.shape[0])
    # np.random.shuffle(reorder)
    # LG_matrix_all = LG_matrix_all[reorder, :]

    # l_f_feat：lncRNA 的高斯相似度特征。
    # g_f_feat：drug 的高斯相似度特征。
    l_f_feat,g_f_feat = gaussian_sim(LG_matrix)  # 46行
    
    
    '''print(l_m_feat.shape)
    print(g_m_feat.shape)
    print(l_f_feat.shape)
    print(g_f_feat.shape)'''
    #print(l_feat.shape)
    #print(g_feat.shape)

    # get_edge_index：构造图中哪些节点之间有边（基于相似度 > 阈值或 KNN）。
    # get_adj：生成邻接矩阵，用于注意力权重的筛选
    Lnc_f_edge_index = get_edge_index(l_f_feat)
    Gene_f_edge_index = get_edge_index(g_f_feat)    
    Lnc_f_adj = get_adj(l_f_feat)
    Gene_f_adj = get_adj(g_f_feat)

    
    
    #adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32) #论文间的邻接矩阵

    # 设置 10 折交叉验证。
    # 每折包含 num 个样本。
    cv=10
    num=int(LG_matrix.shape[0]/cv)

    idx_train = range((cv - 1) * num)
    # idx_val = range((cv-4)*num, (cv-2)*num)
    idx_val = range((cv - 1) * num, LG_matrix.shape[0])
    idx_test = range((cv - 1) * num, LG_matrix.shape[0])

    # 将所有特征矩阵和标签转为 PyTorch 的 Tensor
    Lnc_f_features = torch.FloatTensor(l_f_feat)
    Gene_f_features = torch.FloatTensor(g_f_feat)

    
    LG_matrix = torch.FloatTensor(LG_matrix_all)


    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    idx_train = train_edge_index
    idx_val = val_edge_index
    idx_test = edge_label_index

    # 构建一个字典，用于存储所有数据结构
    dataset = dict()
    #Lnc_adj=Lnc_f_adj+Lnc_m_adj
    #Gene_adj=Gene_f_adj+Gene_m_adj
    
    
    dataset['Lnc_f_edge_index']=Lnc_f_edge_index
    dataset['Lnc_f_adj']=Lnc_f_adj
    dataset['Lnc_f_features']=Lnc_f_features
    dataset['Gene_f_edge_index']=Gene_f_edge_index
    dataset['Gene_f_adj']=Gene_f_adj
    dataset['Gene_f_features']=Gene_f_features
    
    dataset['labels'] = LG_matrix
    dataset['idx_train'] = idx_train
    dataset['idx_val'] = idx_val
    dataset['idx_test'] = idx_test
    return dataset
  
def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def metrics(y_score, y_true):
    y_score=y_score.view(1,-1).detach().cpu().numpy().flatten().tolist()
    y_true=y_true.view(1,-1).detach().cpu().numpy().flatten().tolist()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    y_pred = np.rint(y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auroc=auc(fpr, tpr)
    aupr=auc(recall, precision)
    #acc=accuracy_score(y_true, np.rint(y_score))
    F1=f1_score(y_true, np.rint(y_score), average='macro')
    Pre=precision_score(y_true, np.rint(y_score), average='macro')
    Rec=recall_score(y_true, np.rint(y_score), average='macro')
    Mcc=matthews_corrcoef(y_true,np.rint(y_score))
    acc = accuracy_score(y_true, np.rint(y_score))
    # 计算特异度（Specificity）和灵敏度（Sensitivity）
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sen = tp / (tp + fn)  # 灵敏度 = Recall
    spe = tn / (tn + fp)  # 特异度
    return auroc,aupr,F1,Mcc,acc,Pre,Rec,sen,spe
    
def Auc(y_score, y_true):
    y_score=y_score.view(1,-1).detach().cpu().numpy().flatten().tolist()
    y_true=y_true.view(1,-1).detach().cpu().numpy().flatten().tolist()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auroc=auc(fpr, tpr)
    return auroc

