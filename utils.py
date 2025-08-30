import datetime
import numpy as np
import torch
import random
import seaborn
import os
from sklearn.metrics import roc_curve, roc_auc_score, \
    precision_recall_curve, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt


def get_metrics_auc(real_score, predict_score):
    AUC = roc_auc_score(real_score, predict_score)
    AUPR = average_precision_score(real_score, predict_score)
    return AUC, AUPR


def calculate_metrics(adj_matrix, pred_result):
    y_true = adj_matrix.ravel()
    y_pred = pred_result.ravel()

    auc = roc_auc_score(y_true, y_pred)
    aupr = average_precision_score(y_true, y_pred)

    y_pred_binary = (y_pred >= 0.5).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return auc, aupr, accuracy, f1, precision, recall, specificity


def calculate_metrics_test(edge_index, edge_label, pred_matrix):
    """
    计算二分类评估指标

    参数:
        edge_index (Tensor): 形状 [2, num_edges] 的边索引张量
        edge_label (Tensor): 形状 [num_edges] 的标签 (1=正样本, 0=负样本)
        pred_matrix (Tensor): 形状 [num_drugs, num_diseases] 的预测分数矩阵

    返回:
        tuple: (auc, aupr, accuracy, f1, precision, recall, specificity)
    """

    edge_index = edge_index.cpu().numpy()
    edge_label = edge_label.cpu().numpy()
    pred_matrix = pred_matrix

    drug_ids = edge_index[0]
    disease_ids = edge_index[1]
    y_pred = pred_matrix[drug_ids, disease_ids]


    y_true = edge_label


    auc = roc_auc_score(y_true, y_pred)
    aupr = average_precision_score(y_true, y_pred)


    y_pred_binary = (y_pred >= 0.5).astype(int)


    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()


    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    mcc_numerator = (tp * tn) - (fp * fn)
    mcc_denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = mcc_numerator / mcc_denominator if mcc_denominator != 0 else 0

    return auc, aupr, accuracy, f1, precision, recall, specificity, mcc

def get_metrics(real_score, predict_score):
    """Calculate the performance metrics.
    Resource code is acquired from:
    Yu Z, Huang F, Zhao X et al.
     Predicting drug-disease associations through layer attention graph convolutional network,
     Brief Bioinform 2021;22.

    Parameters
    ----------
    real_score: true labels
    predict_score: model predictions

    Return
    ---------
    AUC, AUPR, Accuracy, F1-Score, Precision, Recall, Specificity
    """
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return auc[0, 0], aupr[0, 0], accuracy, f1_score, precision, recall, specificity


def set_seed(seed=0):
    print('seed = {}'.format(seed))
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    # dgl.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def checkpoint(args, model, check_list, score, fold):
    saved = False
    check_list = np.array(check_list)
    if args.check_metric == 'loss':
        checks = check_list[:, 0]
        if score[0] <= min(checks):
            saved = True
    elif args.check_metric == 'auc':
        checks = check_list[:, 1]
        if score[1] >= max(checks):
            saved = True
    elif args.check_metric == 'aupr':
        checks = check_list[:, 2]
        if score[2] >= max(checks):
            saved = True
    if saved:
        torch.save(model.state_dict(), os.path.join(args.saved_path,
                                                    'model_{}.pkl'.format(fold)))

        # model1 = torch.load('result/B-dataset/k5_topk3_nl2_ep1000_lr0.001_dp0.4_bs2048_hf32_0/'
        #                     f'model_{1}.pkl'
        #                     , map_location='cpu')
        # model11 = torch.load('result/B-dataset/saved_421/'
        #                      f'model_{1}.pkl'
        #                      , map_location='cpu')
        return model


class EarlyStopping(object):
    def __init__(self, patience=10, saved_path='.'):
        dt = datetime.datetime.now()
        self.filename = os.path.join(saved_path, 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second))
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))


def plot_result_auc(args, label, predict, auc):
    """Plot the ROC curve for predictions.
    Parameters
    ----------
    args: argumentation
    label: true labels
    predict: model predictions
    auc: calculated AUROC score
    """
    seaborn.set_style()
    fpr, tpr, threshold = roc_curve(label, predict)
    plt.figure()
    lw = 2
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(args.saved_path, 'result_auc.png'))
    plt.clf()


def plot_result_aupr(args, label, predict, aupr):
    """Plot the ROC curve for predictions.
    Parameters
    ----------
    args: argumentation
    label: true labels
    predict: model predictions
    aupr: calculated AUPR score
    """
    seaborn.set_style()
    precision, recall, thresholds = precision_recall_curve(label, predict)
    plt.figure()
    lw = 2
    plt.figure(figsize=(8, 8))
    plt.plot(precision, recall, color='darkorange',
             lw=lw, label='AUPR Score (area = %0.4f)' % aupr)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('RPrecision/Recall Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(args.saved_path, 'result_aupr.png'))
    plt.clf()
