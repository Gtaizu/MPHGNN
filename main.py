import os

import torch
import pandas as pd
import numpy as np
from warnings import simplefilter

from dgl.nn.pytorch import HeteroGNNExplainer, HeteroPGExplainer, HeteroSubgraphX

from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import TensorDataset

from load_data import load_data, remove_graph, \
    get_data_loaders, load_test_data
from model import Model
from utils import get_metrics, get_metrics_auc, set_seed, \
    plot_result_auc, plot_result_aupr, checkpoint, calculate_metrics, calculate_metrics_test
from args import args
from torch.utils.data import DataLoader

def val(args, model, val_loader, val_label,
        g, feature, device):
    model.eval()
    pred_val = torch.zeros(val_label.shape).to(device)
    with torch.no_grad():
        for i, data_ in enumerate(val_loader):
            x_val, y_val = data_[0].to(device), data_[1].to(device)
            #pred_, attn_, h = model(g, feature, x_val)
            pred_ = model(g, feature, x_val)
            pred_ = pred_.squeeze(dim=1)
            score_ = torch.sigmoid(pred_)
            pred_val[args.batch_size * i: args.batch_size * i + len(y_val)] = score_.detach()
    AUC_val, AUPR_val = get_metrics_auc(val_label.cpu().detach().numpy(), pred_val.cpu().detach().numpy())
    #return AUC_val, AUPR_val, pred_val, h
    return AUC_val, AUPR_val, pred_val

def train():
    simplefilter(action='ignore', category=UserWarning)  # 忽略特定警告
    print('Arguments: {}'.format(args))
    set_seed(args.seed)
    if not os.path.exists(f'result/{args.dataset}'):
        os.makedirs(f'result/{args.dataset}')
    if not os.path.exists(args.saved_path):
        os.makedirs(args.saved_path)

    argsDict = args.__dict__
    with open(os.path.join(args.saved_path, 'setting.txt'), 'w') as f:
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')

    if args.device_id == 'gpu':
        print('Training on GPU')
        device = torch.device('cuda:{}'.format('0'))
    else:
        print('Training on CPU')
        device = torch.device('cpu')


    g, data, label, adj = load_data(args)
    data = torch.tensor(data).to(device)
    label = torch.tensor(label).float().to(device)


    kf = StratifiedKFold(args.nfold, shuffle=True, random_state=args.seed)
    fold = 1

    pred_result = np.zeros((g.num_nodes('miRNA'), g.num_nodes('drug')))

    for (train_idx, val_idx) in kf.split(data.cpu().numpy(), label.cpu().numpy()):
        print('{}-Fold Cross Validation: Fold {}'.format(args.nfold, fold))

        train_data = data[train_idx]
        train_label = label[train_idx]
        val_data = data[val_idx]
        val_label = label[val_idx]
        val_drug_id = [datapoint[0][0].item() for datapoint in val_data]
        val_disease_id = [datapoint[0][-1].item() for datapoint in val_data]

        dda_idx = torch.where(val_label == 1)[0].cpu().numpy()
        val_dda_drugid = np.array(val_drug_id)[dda_idx]
        val_dda_disid = np.array(val_disease_id)[dda_idx]

        g_train = g
        g_train = remove_graph(g_train, val_dda_drugid.tolist(), val_dda_disid.tolist()).to(device)
        feature = {'miRNA': g_train.nodes['miRNA'].data['h'],
                   'drug': g_train.nodes['drug'].data['h'],
                   'gene': g_train.nodes['gene'].data['h']}
        train_loader = get_data_loaders(TensorDataset(train_data, train_label), args.batch_size,
                                        shuffle=True, drop=True)
        val_loader = get_data_loaders(TensorDataset(val_data, val_label), args.batch_size, shuffle=False)

        model = Model(g.etypes,
                      {'miRNA': feature['miRNA'].shape[1], 'drug': feature['drug'].shape[1],
                       'gene': feature['gene'].shape[1]},
                      hidden_feats=args.hidden_feats,
                      num_emb_layers=args.num_layer,
                      agg_type=args.aggregate_type,
                      dropout=args.dropout,
                      bn=args.batch_norm,
                      k=args.topk)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate,
                                     weight_decay=args.weight_decay)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(len(torch.where(train_label == 0)[0]) /
                                                                       len(torch.where(train_label == 1)[0])))
        print('BCE loss pos weight: {:.4f}'.format(
            len(torch.where(train_label == 0)[0]) / len(torch.where(train_label == 1)[0])))

        record_list = []
        print_list = []

        for epoch in range(1, args.epoch + 1):
            total_loss = 0

            pred_train, label_train = torch.zeros(train_label.shape).to(device), \
                                      torch.zeros(train_label.shape).to(device)
            for i, data_ in enumerate(train_loader):
                model.train()
                x_train, y_train = data_[0].to(device), data_[1].to(device)
                #pred, attn, h = model(g_train, feature, x_train)
                pred = model(g_train, feature, x_train)
                pred = pred.squeeze(dim=1)
                score = torch.sigmoid(pred)
                optimizer.zero_grad()
                loss = criterion(pred, y_train)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() / len(train_loader)
                # progress.set_description("Loss: {:.4f}".format(total_loss / (i + 1)))
                pred_train[args.batch_size * i: args.batch_size * i + len(y_train)] = score.detach()
                label_train[args.batch_size * i: args.batch_size * i + len(y_train)] = y_train.detach()

            AUC_train, AUPR_train = get_metrics_auc(label_train.cpu().detach().numpy(),
                                                    pred_train.cpu().detach().numpy())

            # if epoch % args.print_every == 0:
            #AUC_val, AUPR_val, pred_val, h = val(args, model, val_loader, val_label, g_train, feature, device)
            AUC_val, AUPR_val, pred_val = val(args, model, val_loader, val_label, g_train, feature, device)
            if epoch % args.print_every == 0:
                print('Epoch {} Loss: {:.5f}; Train: AUC {:.4f}, AUPR {:.4f};'
                      ' Val: AUC {:.4f}, AUPR {:.4f}'.format(epoch, total_loss, AUC_train,
                                                             AUPR_train, AUC_val, AUPR_val))
                record_list.append([total_loss, AUC_train, AUPR_train, AUC_val, AUPR_val])
            print_list.append([total_loss, AUC_train, AUPR_train])
            m = checkpoint(args, model, print_list, [total_loss, AUC_train, AUPR_train], fold)
            if m:
                best_model = m
            # print('Epoch {} Loss: {:.3f}; Train: AUC {:.3f}, AUPR {:.3f}'.format(epoch, total_loss,
            #                                                                      AUC_train, AUPR_train))

        #AUC_val, AUPR_val, pred_val, h = val(args, best_model, val_loader, val_label, g_train, feature, device)
        AUC_val, AUPR_val, pred_val = val(args, best_model, val_loader, val_label, g_train, feature, device)
        pred_result[val_drug_id, val_disease_id] = pred_val.cpu().detach().numpy()
        print("record_list content:", record_list)

        pd.DataFrame(np.array(record_list),
                     columns=['Loss', 'AUC_train', 'AUPR_train',
                              'AUC_val', 'AUPR_val']).to_csv(os.path.join(args.saved_path,
                                                                          'training_score_{}.csv'.format(fold)),
                                                             index=False)
        fold += 1
        # break
    Pred_test = np.zeros((g.num_nodes('miRNA'), g.num_nodes('drug')))
    g_test, data_test, label_test, edge, edge_label = load_test_data(args)

    g_test = g_test.to(device)
    data_test = torch.tensor(data_test).to(device)
    label_test = torch.tensor(label_test).float().to(device)
    test_drug_id = [datapoint[0][0].item() for datapoint in data_test]
    test_disease_id = [datapoint[0][-1].item() for datapoint in data_test]
    feature = {'miRNA': g_test.nodes['miRNA'].data['h'],
               'drug': g_test.nodes['drug'].data['h'],
               'gene': g_test.nodes['gene'].data['h']}

    test_loader = DataLoader(TensorDataset(data_test, label_test), batch_size=len(data_test), shuffle=False)

    #Auc_test, AUPR_test, pred_test, h = val(args, best_model, test_loader, label_test, g_test, feature, device)
    Auc_test, AUPR_test, pred_test = val(args, best_model, test_loader, label_test, g_test, feature, device)
    Pred_test[test_drug_id, test_disease_id] = pred_test.cpu().detach().numpy()
    AUC, AUPR, Acc, F1, Pre, Rec, Spec, mcc = calculate_metrics_test(edge, edge_label, Pred_test)
    print('Overall: AUC {:.4f}, AUPR: {:.4f}, Accuracy: {:.4f},'
          ' F1 {:.4f}, Precision {:.4f}, Recall {:.4f}, Specificity {:.4f}, MCC {:.4f}'.format(
        AUC, AUPR, Acc, F1, Pre, Rec, Spec, mcc))



if __name__ == '__main__':
    train()
