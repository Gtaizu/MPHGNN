import os
from parameters import args_parser
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from model import Model
from subgraph_extraction import links2subgraphs
from utils import *
from prediction import preds
import warnings

warnings.filterwarnings('ignore')
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def cv_process():
    args = args_parser()
    if not os.path.exists('result/'):
        os.mkdir('result/')
    data = np.load('../data/split_data_' + args.dataset + '.npz', allow_pickle=True)
    # data1 = data['md']
    # idx_train_val = data['idx_train_val']
    # idx_test = data['idx_test']
    kfolds = data['kfolds']
    # kfs = data['kfs']

    data = torch.load("ncRNADrug2.pth", weights_only=False)
    edge_index = data['ncRNA', 'ncDrug', 'drug'].edge_index
    miRNA_drug = data['ncRNA', 'ncDrug', 'drug'].edge_label_index
    miRNA_drug_label = data['ncRNA', 'ncDrug', 'drug'].edge_label

    test_samples = np.column_stack((
        miRNA_drug[0].cpu().numpy(),  # miRNA 索引
        miRNA_drug[1].cpu().numpy(),  # drug 索引
        miRNA_drug_label.cpu().numpy()  # 标签 (1 或 0)
    ))

    positive_samples = np.column_stack((
        edge_index[0].cpu().numpy(),
        edge_index[1].cpu().numpy(),
        np.ones(edge_index.shape[1])
    ))

    # 2. 生成负样本
    num_positive = positive_samples.shape[0]
    num_mirna = data['ncRNA'].num_nodes
    num_drug = data['drug'].num_nodes
    existing_edges = set(zip(edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()))

    negative_samples = []
    while len(negative_samples) < num_positive:
        mirna = np.random.randint(0, num_mirna)
        drug = np.random.randint(0, num_drug)
        if (mirna, drug) not in existing_edges:
            negative_samples.append([mirna, drug, 0])
    negative_samples = np.array(negative_samples)

    # 3. 合并并打乱
    all_samples = np.vstack((positive_samples, negative_samples))
    np.random.shuffle(all_samples)
    # 4. 处理测试集
    test_edges = miRNA_drug.cpu().numpy().T

    train_samples = []
    for sample in all_samples:
        if (sample[0], sample[1]) not in set(zip(test_edges[:, 0], test_edges[:, 1])):
            train_samples.append(sample)
    train_samples = np.array(train_samples)

    idx_train_val = train_samples
    idx_test = test_samples

    # 假设 train_samples 是平衡后的训练集，形状为 [num_samples, 3]
    n_splits = 5  # 5 折
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)  # 设置随机种子确保可复现

    # 存储每一折的训练集和验证集索引
    kfs = []
    for train_idx, val_idx in kf.split(train_samples):
        kfs.append((train_idx, val_idx))  # 存入 kfs 列表

    # 获取 miRNA 和 drug 的数量
    num_mirna = data['ncRNA'].num_nodes
    num_drug = data['drug'].num_nodes

    # 初始化全零矩阵
    data1 = np.zeros((num_mirna, num_drug), dtype=int)

    for i in range(edge_index.shape[1]):
        mirna = edge_index[0, i].item()
        drug = edge_index[1, i].item()
        data1[mirna, drug] = 1

    # --- 2. 填充测试集的正样本 ---
    for i in range(miRNA_drug.shape[1]):
        if miRNA_drug_label[i] == 1:  # 仅处理正样本
            mirna = miRNA_drug[0, i].item()
            drug = miRNA_drug[1, i].item()
            data1[mirna, drug] = 1

    n_classes = np.unique(data1).size

    test_positive_edges = np.array([[int(i[0]), int(i[1])] for i in idx_test if i[2] == 1])  # 正样本
    test_positive_edges2 = np.array([[int(i[1]), int(i[0])] for i in idx_test if i[2] == 1])
    test_zero_edges = np.array([[int(i[0]), int(i[1])] for i in idx_test if i[2] == 0])  # 负样本

    test_positive_edges = (test_positive_edges.T[0], test_positive_edges.T[1])  # 转换为元组格式
    test_positive_edges2 = (test_positive_edges2.T[0], test_positive_edges2.T[1])
    test_zero_edges = (test_zero_edges.T[0], test_zero_edges.T[1])
    # 默认 5 折交叉验证
    for iters in range(5):
        idx_train = kfs[iters][0]
        idx_val = kfs[iters][1]

        train_edges = [idx_train_val[i] for i in idx_train]

        val_edges = [idx_train_val[i] for i in idx_val]
        val_positive_edges = np.array([[int(i[0]), int(i[1])] for i in val_edges if i[2] == 1])
        val_positive_edges2 = np.array([[int(i[1]), int(i[0])] for i in val_edges if i[2] == 1])
        val_zero_edges = np.array([[int(i[0]), int(i[1])] for i in val_edges if i[2] == 0])

        val_positive_edges = (val_positive_edges.T[0], val_positive_edges.T[1])
        val_positive_edges2 = (val_positive_edges2.T[0], val_positive_edges2.T[1])
        val_zero_edges = (val_zero_edges.T[0], val_zero_edges.T[1])

        MD = np.copy(data1)
        MD[test_positive_edges] = 0  # 掩码测试集正样本
        MD[val_positive_edges] = 0  # 掩码验证集正样本
        # 如果是同质图（无向图）
        if args.types == 'homo':
            MD[test_positive_edges2] = 0  # 掩码反向边
            MD[val_positive_edges2] = 0
            MD2 = np.tril(MD, k=0)  # 取下三角矩阵（避免重复边）
        else:
            MD2 = MD
        train_positive_edges = np.where(MD2 == 1)
        train_zero_edges = np.array(
            [[int(i[0]), int(i[1])] for i in train_edges if i[2] == 0])  # [:len(train_positive_edges[0]),:]
        train_zero_edges = (train_zero_edges.T[0], train_zero_edges.T[1])

        print(len(train_positive_edges[0]), len(train_zero_edges[0]), len(val_positive_edges[0]),
              len(val_zero_edges[0]))

        if args.types == 'homo':
            MD2 = MD2 + MD2.T
            featM, featD, featM_sim, featD_sim = comp_feat2(MD2, args.dataset)  # utils
        else:
            featM, featD, featM_sim, featD_sim = comp_feat(MD2, args.dataset)

        train_graphs, val_graphs, _, max_n = links2subgraphs(MD2, train_positive_edges, train_zero_edges,
                                                             val_positive_edges, val_zero_edges,
                                                             None, None,
                                                             h=1, max_nodes_per_hop=None, featM=featM, featD=featD)
        n_train = len(train_graphs)
        n_val = len(val_graphs)

        if args.fixs == 0:
            size_subgraphs = int(np.mean([ii[0].shape[0] for ii in train_graphs]))
        else:
            size_subgraphs = args.fixs
        size_graph_filters = [int(size_subgraphs)]

        # Sampling
        adj_train = [i[0] for i in train_graphs]
        nodes_trains = [i[2] for i in train_graphs]
        y_train = [i[3] for i in train_graphs]

        adj_val = [i[0] for i in val_graphs]
        nodes_val = [i[2] for i in val_graphs]
        y_val = [i[3] for i in val_graphs]

        features_train = [i[1] for i in train_graphs]
        features_val = [i[1] for i in val_graphs]

        features_dim = features_train[0].shape[1]

        # Create model
        model = Model(featM_sim.shape[1], featD_sim.shape[1],
                      features_dim, n_classes, args,
                      max_step=args.max_step, dropout_rate=args.dropout_rate,
                      size_graph_filter=size_graph_filters)
        #print(model)
        model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        cv_trains = []
        cv_vals = []
        losses_train = []
        losses_val = []
        best_score = 0
        counter = 0

        for epoch in range(args.epochs):
            model.train()
            loss_tra, output_tra, targets_tra, cv_tra = preds(MD2, adj_train, features_train, nodes_trains, y_train,
                                                              featM_sim, featD_sim, args.batch_size, epoch, model,
                                                              optimizer=optimizer, criterion=criterion)
            print('epoch: ', epoch, loss_tra, '\n', cv_tra)
            cv_trains.append(cv_tra)
            losses_train.append(loss_tra)

            model.eval()
            loss_val, output_val, targets_val, cv_val = preds(MD2, adj_val, features_val, nodes_val, y_val,
                                                              featM_sim, featD_sim, args.batch_size, epoch, model,
                                                              optimizer=None, criterion=criterion)
            print('epoch: ', epoch, loss_val, '\n', cv_val)
            cv_vals.append(cv_val)
            losses_val.append(loss_val)

            if best_score > cv_val[3]:
                counter += 1
                if counter >= 20:
                    break
            else:
                best_score = cv_val[3]
                counter = 0
        np.savetxt('result/kf_trains_' + str(args.dataset) + '_iters_' + str(iters) + '.txt', np.array(cv_trains))
        np.savetxt('result/kf_validations_' + str(args.dataset) + '_iters_' + str(iters) + '.txt', np.array(cv_vals))

    # 初始化全零矩阵
    data1 = np.zeros((num_mirna, num_drug), dtype=int)

    # --- 2. 填充测试集的正样本 ---
    for i in range(miRNA_drug.shape[1]):
        if miRNA_drug_label[i] == 1:  # 仅处理正样本
            mirna = miRNA_drug[0, i].item()
            drug = miRNA_drug[1, i].item()
            data1[mirna, drug] = 1

    n_classes = np.unique(data1).size
    MD = np.copy(data1)

    MD2 = MD
    featM, featD, featM_sim, featD_sim = comp_feat(MD2, args.dataset)
    _, _, test_graphs, max_n = links2subgraphs(MD2, None, None,
                                               None, None,
                                               test_positive_edges, test_zero_edges,
                                               h=1, max_nodes_per_hop=None, featM=featM, featD=featD)

    adj_test = [i[0] for i in test_graphs]
    nodes_test = [i[2] for i in test_graphs]
    y_test = [i[3] for i in test_graphs]
    features_test = [i[1] for i in test_graphs]
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_val, output_val, targets_val, cv_val = preds(MD2, adj_test, features_test, nodes_test, y_test,
                                                      featM_sim, featD_sim, args.batch_size, 1, model,
                                                      optimizer=None, criterion=criterion)

    print('epoch: ', 1, loss_val, '\n', cv_val)
    print(output_val)
    print(targets_val)

    scores = np.array(output_val)[:, 1]  # 确保 outputs1 是 NumPy 数组
    miRNA_indices = miRNA_drug[0].cpu().numpy()  # miRNA 索引数组
    drug_indices = miRNA_drug[1].cpu().numpy()  # drug 索引数组

    # 组合成 miRNA-drug 对
    miRNA_drug_pairs = list(zip(miRNA_indices, drug_indices))
    # 2. 构建 DataFrame
    results_df = pd.DataFrame({
        'ncRNAid': [f"{pair[0]}" for pair in miRNA_drug_pairs],  # 替换为实际 miRNA ID
        'drugid': [f"{pair[1]}" for pair in miRNA_drug_pairs],  # 替换为实际 drug ID
        'score': scores,
        'label': targets_val
    })

    # 3. 保存到 CSV
    results_df.to_csv('ncRNA_drug_predictions.csv', index=False)
    print("预测结果已保存到 miRNA_drug_predictions.csv")
