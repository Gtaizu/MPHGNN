from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import parameter_parser,load_data, metrics,Auc,get_edge_index
from models import Deep_LDA


def negative_sampling(labels, pos_edges, test_edges, num_neg=1):
    """
    更高效的负采样方法，避免训练/测试正样本重叠
    labels: [num_lnc, num_drug] 0/1 标签张量
    pos_edges, test_edges: 正样本索引对
    num_neg: 每个正样本采样几个负样本
    """
    device = labels.device
    num_lnc, num_drug = labels.shape
    num_pos = len(pos_edges[0])

    # 构建禁止采样的索引掩码
    block_mask = torch.zeros_like(labels, dtype=torch.bool)
    block_mask[pos_edges[0], pos_edges[1]] = True
    block_mask[test_edges[0], test_edges[1]] = True

    # 创建候选负样本索引矩阵
    candidate_mask = ~block_mask  # 可采样区域为 True
    candidate_idx = candidate_mask.nonzero(as_tuple=False)  # [N, 2]

    # 从候选中随机采样
    num_needed = num_pos * num_neg
    rand_idx = torch.randperm(candidate_idx.shape[0], device=device)[:num_needed]
    sampled_pairs = candidate_idx[rand_idx]  # shape [num_needed, 2]

    lnc_idx_neg = sampled_pairs[:, 0]
    drug_idx_neg = sampled_pairs[:, 1]

    return lnc_idx_neg, drug_idx_neg


def train(model,args,dataset):
    # 使用 Adam 优化器训练
    optimizer = torch.optim.Adam(model.parameters(),
                           lr=args.lr, 
                           weight_decay=args.weight_decay)                        
    model.train()
    t_total = time.time()
    loss_values = []
    total_auc = []
    total_aupr = []
    total_f1 = []
    total_mcc = []
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = 0
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        Lnc_output, x = model.forward(dataset)
        output = Lnc_output
        # 正样本
        lnc_idx_pos = dataset['idx_train'][0]
        drug_idx_pos = dataset['idx_train'][1]
        lnc_idx_test = dataset['idx_test'][0]
        drug_idx_test = dataset['idx_test'][1]
        labels = dataset['labels']
        true_pos = torch.ones_like(lnc_idx_pos).float().to(output.device)

        # 负样本采样
        lnc_idx_neg, drug_idx_neg = negative_sampling(labels, (lnc_idx_pos, drug_idx_pos),(lnc_idx_test, drug_idx_test))
        lnc_idx_neg = lnc_idx_neg.to(output.device)
        drug_idx_neg = drug_idx_neg.to(output.device)
        true_neg = torch.zeros_like(lnc_idx_neg).float().to(output.device)
        # 拼接训练样本
        lnc_idx_all = torch.cat([lnc_idx_pos, lnc_idx_neg])
        drug_idx_all = torch.cat([drug_idx_pos, drug_idx_neg])
        y_true = torch.cat([true_pos, true_neg])
        y_pred = output[lnc_idx_all, drug_idx_all]

        # 损失函数可以换成 BCE（二分类）
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss_train = loss_fn(y_pred, y_true)
        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            Lnc_output, x = model.forward(dataset)
            output = Lnc_output
        lnc_idx = dataset['idx_val'][0]
        drug_idx = dataset['idx_val'][1]
        pred = output[lnc_idx, drug_idx]
        true = labels[lnc_idx, drug_idx]
        loss_val = loss_fn(pred, true)
        auc_val = Auc(pred, true)
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.8f}'.format(loss_train.data.item()),
              #'auc_train: {:.4f}'.format(auc_train),
              'loss_val: {:.8f}'.format(loss_val.data.item()),
              #'auc_val: {:.4f}'.format(auc_val),
              #'time: {:.4f}s'.format(time.time() - t)
              )

    #np.savetxt(args.result_path+'total_auc.txt', total_auc,delimiter='\t')   #x,y,z相同大小的一维数组
    #np.savetxt(args.result_path+'total_aupr.txt', total_aupr,delimiter='\t')   #x,y,z相同大小的一维数组
    #np.savetxt(args.result_path+'total_f1.txt', total_f1,delimiter='\t')   #x,y,z相同大小的一维数组
    #np.savetxt(args.result_path+'total_mcc.txt', total_mcc,delimiter='\t')   #x,y,z相同大小的一维数组
    '''
        loss_values.append(loss_val.data.item())
        torch.save(model.state_dict(), '{}.pkl'.format(epoch))
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1
        #if bad_counter == args.patience:
        #    break
        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)
    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)
    
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
    '''
  
def compute_test(model,dataset):
    model.eval()
    Lnc_output, x = model.forward(dataset)
    output = Lnc_output
    lnc_idx = dataset['idx_test'][0]
    drug_idx = dataset['idx_test'][1]
    lnc_idx1 = dataset['idx_train'][0]
    drug_idx1 = dataset['idx_train'][1]
    labels = dataset['labels']
    pred1 = output[lnc_idx1, drug_idx1]
    true1 = labels[lnc_idx1, drug_idx1]
    pred = output[lnc_idx, drug_idx]
    true = labels[lnc_idx, drug_idx]
    loss = torch.nn.MSELoss(reduction='mean')
    loss_test = loss(pred, true)
    #auc_train,aupr_train,F1_train,mcc_train = metrics(pred1, true1)
    auc_val,aupr_val,F1_val,mcc_val,acc_val,pre_val,rec_val,sen_val,spe_val = metrics(pred, true)
    auc_test,aupr_test,F1_test,mcc_test,acc_test,pre_test,rec_test,sen_test,spe_test = metrics(pred, true)

    # print("Train set results:",
    #       "auc_train= {:.4f}".format(auc_train),
    #       "aupr_train= {:.4f}".format(aupr_train),
    #       #"acc_train= {:.4f}".format(acc_train),
    #       "F1_train= {:.4f}".format(F1_train),
    #       #"Pre_train= {:.4f}".format(Pre_train),
    #       #"Rec_train= {:.4f}".format(Rec_train)
    #       "mcc_train= {:.4f}".format(mcc_train)
    #       )
    print("Val set results:",
          "auc_val= {:.4f}".format(auc_val),
          "aupr_val= {:.4f}".format(aupr_val),
          "acc_val= {:.4f}".format(acc_val),
          "F1_val= {:.4f}".format(F1_val),
          "Pre_val= {:.4f}".format(pre_val),
          "Rec_val= {:.4f}".format(rec_val),
          "mcc_val= {:.4f}".format(mcc_val),
          "acc_val= {:.4f}".format(acc_val)
          )
    print("Test set results:",
          #"loss_test= {:.4f}".format(loss_test.data.item()),
          "auc_test= {:.4f}".format(auc_test),
          "aupr_test= {:.4f}".format(aupr_test),
          "acc_test= {:.4f}".format(acc_test),
          "F1_test= {:.4f}".format(F1_test),
          "Pre_test= {:.4f}".format(pre_test),
          "Rec_test= {:.4f}".format(rec_test),
          "mcc_test= {:.4f}".format(mcc_test),
          "acc_test= {:.4f}".format(acc_test),
          "sen_test= {:.4f}".format(sen_test),
          "spe_test= {:.4f}".format(spe_test)
          )
    data = torch.load("ncRNADrug2.pth", weights_only=False)
    edge_label_index = data[("ncRNA", "ncDrug", "drug")].edge_label_index
    edge_label = data[("ncRNA", "ncDrug", "drug")].edge_label
    # 找到标签为 1 的边的索引位置
    pos_indices = (edge_label == 1).nonzero(as_tuple=True)[0]
    neg_indices = (edge_label == 0).nonzero(as_tuple=True)[0]
    # 提取对应的边（正样本边）
    pos_edge_index = edge_label_index[:, pos_indices]
    neg_edge_index = edge_label_index[:, neg_indices]
    all_edges = dataset['idx_test']  # shape: [2, num_edges]
    all_labels = torch.cat([torch.ones(pos_edge_index.size(1)), torch.zeros(neg_edge_index.size(1))], dim=0)
    num_miRNA = data['ncRNA'].num_nodes
    src_nodes = all_edges[0].cpu().numpy()
    dst_nodes = all_edges[1].cpu().numpy()
    scores = pred.detach().cpu().numpy()
    labels = all_labels.cpu().numpy()

    print(src_nodes.shape)
    print(dst_nodes.shape)
    print(scores.shape)
    print(labels.shape)
    # 构建 DataFrame
    df = pd.DataFrame({
        'ncRNA_id': src_nodes,
        'drug_id': dst_nodes,
        'score': scores,
        'label': labels
    })
    # 保存到文件
    df.to_csv('ncRNA_drug_prediction_results.csv', index=False)

def main():
    # Training settings
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    args = parameter_parser()  # utils 9行
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # 为 Python 的随机模块、NumPy 和 PyTorch 设置固定随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # Load data
    dataset = load_data(args)  # utils 82行
    # Model and optimizer
    # 读取 lncRNA 和 Gene 的输入特征维度
    args.l_f_nfeat=dataset['Lnc_f_features'].shape[1]
    args.g_f_nfeat=dataset['Gene_f_features'].shape[1]
    # args.nclass=dataset['Gene_f_features'].shape[1]

    # 创建模型实例
    model = Deep_LDA(args)  # model 7行
    if args.cuda:
        model.cuda()
        dataset['Lnc_f_edge_index'] = dataset['Lnc_f_edge_index'].cuda()
        dataset['Lnc_f_adj'] = dataset['Lnc_f_adj'].cuda()
        dataset['Lnc_f_features'] = dataset['Lnc_f_features'].cuda()
        dataset['Gene_f_edge_index'] = dataset['Gene_f_edge_index'].cuda()
        dataset['Gene_f_adj'] = dataset['Gene_f_adj'].cuda()
        dataset['Gene_f_features'] = dataset['Gene_f_features'].cuda()    
        dataset['labels'] = dataset['labels'].cuda()
        dataset['idx_train'] = dataset['idx_train'].cuda()
        dataset['idx_val'] = dataset['idx_val'].cuda()
        dataset['idx_test'] = dataset['idx_test'].cuda()
        #dataset['Lnc_f_edge_index'], dataset['Lnc_f_adj'], dataset['Lnc_f_features'], dataset['Gene_f_edge_index'], dataset['Gene_f_adj'], dataset['Gene_f_features'], dataset['labels'] = Variable(dataset['Lnc_f_edge_index']), Variable(dataset['Lnc_f_adj']), Variable(dataset['Lnc_f_features']), Variable(dataset['Gene_f_edge_index']), Variable(dataset['Gene_f_adj']), Variable(dataset['Gene_f_features']), Variable(dataset['labels'])
    # Train model
    train(model,args,dataset)  # 16行
    # Testing
    compute_test(model,dataset)

if __name__ == "__main__":
    main()
