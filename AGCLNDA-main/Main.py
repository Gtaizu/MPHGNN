import pandas as pd
import torch
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args# 参数配置文件
from Model import AGCLNDA, vgae_encoder, vgae_decoder, vgae, DenoisingNet# 模型组件
from DataHandler import DataHandler# 数据处理器
import numpy as np
from Utils.Utils import calcRegLoss, pairPredict  # 工具函数
import os
from copy import deepcopy  # 深度拷贝工具
import scipy.sparse as sp  # 稀疏矩阵处理
import random
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, \
    recall_score, f1_score, matthews_corrcoef, confusion_matrix  # 评估指标
# 启用CUDA加速配置
torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True

class Coach:
    def __init__(self, handler):
        self.handler = handler  # 数据处理器实例

        print('nc', args.nc, 'drug', args.drug)
        print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()  # 存储评估指标

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        self.prepareModel()  # 初始化模型和优化器
        log('Model Prepared')
        best_auc = 0  # 记录最佳AUC
        best_aupr = 0  # 记录最佳AUPR
        stloc = 0  # 起始epoch
        log('Model Initialized')

        # 训练循环
        for ep in range(stloc, args.epoch):
            temperature = max(0.05, args.init_temperature * pow(args.temperature_decay, ep))
            tstFlag = (ep % args.tstEpoch == 0)  # 是否进行测试
            reses = self.trainEpoch(temperature)  # 训练一个epoch
            log(self.makePrint('Train', ep, reses, tstFlag))
            if tstFlag:  # 测试阶段
                y_true, y_pred, pred_score = self.testEpoch()
                y_pred1 = np.array(y_pred)
                y_pred_binary = (y_pred1 > 0.5).astype(int)
                auc_value = roc_auc_score(y_true, y_pred)
                aupr_value = average_precision_score(y_true, y_pred)
                acc = accuracy_score(y_true, y_pred_binary)
                pre = precision_score(y_true, y_pred_binary)
                sen = recall_score(y_true, y_pred_binary)  # 灵敏度=召回率
                f1 = f1_score(y_true, y_pred_binary)
                mcc = matthews_corrcoef(y_true, y_pred_binary)

                # 计算特异度
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
                spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0

                # 更新最佳指标
                if auc_value > best_auc:
                    best_auc = auc_value
                    best_aupr = aupr_value
                    best_acc = acc
                    best_pre = pre
                    best_sen = sen
                    best_spe = spe
                    best_f1 = f1
                    best_mcc = mcc

                # 打印当前epoch指标
                print(f"Epoch {ep}: "
                      f"AUC={auc_value:.4f}, AUPR={aupr_value:.4f}, "
                      f"Acc={acc:.4f}, Pre={pre:.4f}, Sen={sen:.4f}, "
                      f"Spe={spe:.4f}, F1={f1:.4f}, MCC={mcc:.4f}")

        # 输出最终最佳结果
        print("\nBest Results: "
            f"AUC={best_auc:.4f}, AUPR={best_aupr:.4f}, "
            f"Acc={best_acc:.4f}, Pre={best_pre:.4f}, Sen={best_sen:.4f}, "
            f"Spe={best_spe:.4f}, F1={best_f1:.4f}, MCC={best_mcc:.4f}")

    def prepareModel(self):
        """初始化模型和优化器"""
        # 主模型
        self.model = AGCLNDA().cuda()  # Model 23行
        # 变分图自编码器组件
        encoder = vgae_encoder().cuda()
        decoder = vgae_decoder().cuda()
        self.generator_1 = vgae(encoder, decoder).cuda()

        # 去噪网络 接收主模型的 GCN 网络和当前节点嵌入作为输入
        self.generator_2 = DenoisingNet(self.model.getGCN(), self.model.getEmbeds()).cuda()
        # getGCN Model 117行 DenoisingNet 222行 getEmbeds Model 106行
        # set_fea_adj Model 358行
        # 设置去噪网络的输入图结构：
        # torchBiAdj 是一个稀疏的二部图邻接矩阵（ncRNA + drug）；
        # deepcopy 是为了保证该图在 generator_2 内部使用过程中不会被修改；
        self.generator_2.set_fea_adj(args.nc+args.drug, deepcopy(self.handler.torchBiAdj).cuda())

        # 优化器配置
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        self.opt_gen_1 = torch.optim.Adam(self.generator_1.parameters(), lr=args.lr, weight_decay=0)
        self.opt_gen_2 = torch.optim.Adam(filter(lambda p: p.requires_grad, self.generator_2.parameters()), lr=args.lr, weight_decay=0, eps=args.eps)

    def trainEpoch(self, temperature):
        """单个训练epoch，预测优化模块"""
        trnLoader = self.handler.trnLoader
        trnLoader.dataset.negSampling()  # 负采样
        generate_loss_1, generate_loss_2, bpr_loss, im_loss, ib_loss, reg_loss = 0, 0, 0, 0, 0, 0
        steps = trnLoader.dataset.__len__() // args.batch

        for i, tem in enumerate(trnLoader):
            data = deepcopy(self.handler.torchBiAdj).cuda()  # 获取邻接矩阵
            # 生成增强视图
            data1 = self.generator_generate(self.generator_1)
            # 梯度清零
            self.opt.zero_grad()
            self.opt_gen_1.zero_grad()
            self.opt_gen_2.zero_grad()
            # 获取批次数据
            ancs, poss, negs = tem
            ancs = ancs.long().cuda()
            poss = poss.long().cuda()
            negs = negs.long().cuda()
            # 对比学习损失计算
            out1 = self.model.forward_graphcl(data1)
            out2 = self.model.forward_graphcl_(self.generator_2)

            loss = self.model.loss_graphcl(out1, out2, ancs, poss).mean() * args.ssl_reg
            im_loss += float(loss)
            loss.backward()

            self.opt.step()
            self.opt.zero_grad()

            # info bottleneck
            # 信息瓶颈正则化
            _out1 = self.model.forward_graphcl(data1)
            _out2 = self.model.forward_graphcl_(self.generator_2)

            loss_ib = self.model.loss_graphcl(_out1, out1.detach(), ancs, poss) + self.model.loss_graphcl(_out2, out2.detach(), ancs, poss)
            loss = loss_ib.mean() * args.ib_reg
            ib_loss += float(loss)
            loss.backward()

            self.opt.step()
            self.opt.zero_grad()

            # BPR
            # BPR损失计算
            ncEmbeds, drugEmbeds = self.model.forward_gcn(data)
            ancEmbeds = ncEmbeds[ancs]
            posEmbeds = drugEmbeds[poss]
            negEmbeds = drugEmbeds[negs]
            scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
            bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch
            regLoss = calcRegLoss(self.model) * args.reg
            loss = bprLoss + regLoss
            bpr_loss += float(bprLoss)
            reg_loss += float(regLoss)
            loss.backward()
            # 生成器损失
            loss_1 = self.generator_1(deepcopy(self.handler.torchBiAdj).cuda(), ancs, poss, negs)
            loss_2 = self.generator_2(ancs, poss, negs, temperature)

            loss = loss_1 + loss_2
            generate_loss_1 += float(loss_1)
            generate_loss_2 += float(loss_2)
            loss.backward()
            # 参数更新
            self.opt.step()
            self.opt_gen_1.step()
            self.opt_gen_2.step()
            # 打印进度
            log('Step %d/%d: gen 1 : %.3f ; gen 2 : %.3f ; bpr : %.3f ; im : %.3f ; ib : %.3f ; reg : %.3f  ' % (
                i,
                steps,
                generate_loss_1,
                generate_loss_2,
                bpr_loss,
                im_loss,
                ib_loss,
                reg_loss,
                ), save=False, oneline=True)
        # 计算平均损失
        ret = dict()
        ret['Gen_1 Loss'] = generate_loss_1 / steps
        ret['Gen_2 Loss'] = generate_loss_2 / steps
        ret['BPR Loss'] = bpr_loss / steps
        ret['IM Loss'] = im_loss / steps
        ret['IB Loss'] = ib_loss / steps
        ret['Reg Loss'] = reg_loss / steps

        return ret

    # def testEpoch(self):
    #     """测试epoch"""
    #     tstLoader = self.handler.tstLoader
    #     i = 0
    #     num = tstLoader.dataset.__len__()
    #     test_labels = self.handler.tstData.tstLocs
    #     train_labels = self.handler.trnData.trnLocs
    #     y_pred = []
    #     y_true = []
    #     steps = num // args.tstBat
    #     for nc, trnMask in tstLoader:
    #         i += 1
    #         nc = nc.long().cuda()
    #         trnMask = trnMask.cuda()
    #         # 获取预测分数
    #         ncEmbeds, drugEmbeds = self.model.forward_gcn(self.handler.torchBiAdj)
    #         pred_score = torch.mm(ncEmbeds[nc], torch.transpose(drugEmbeds, 1, 0))
    #         pred_score_all = torch.mm(ncEmbeds, torch.transpose(drugEmbeds, 1, 0))
    #         # 转换为numpy数组
    #         pred_score = pred_score.detach().cpu().numpy().tolist()
    #         pred_score_all = pred_score_all.detach().cpu().numpy().tolist()
    #
    #         data = torch.load("MiDrug_test_data.pth", weights_only=False)
    #         edge_label = data[("miRNA", "MiDrug", "drug")].edge_label
    #         edge_label_index = data[("miRNA", "MiDrug", "drug")].edge_label_index
    #         neg_indices = (edge_label == 0).nonzero(as_tuple=True)[0]
    #         neg_index = edge_label_index[:, neg_indices]
    #
    #         # 处理每个样本
    #         for i in range(len(nc)):
    #             ncid = nc[i]
    #             drug_scores = pred_score[i]
    #             pos = test_labels[ncid]# 真实正样本
    #             train_drug_ass = train_labels[ncid]  # 训练集正样本
    #             # 生成负样本
    #             if pos is None:
    #                 pos = []
    #             if train_drug_ass is None:
    #                 train_drug_ass = []
    #             train_test_ass = pos + train_drug_ass
    #
    #             # diff = list(set(range(len(drug_scores))) - set(train_test_ass))
    #             # random.shuffle(diff)
    #             # neg = diff[0:len(pos)]
    #             # y_true += [1] * len(pos)
    #             # y_true += [0] * len(pos)
    #             # for drug in pos:
    #             #     y_pred.append(drug_scores[drug])
    #             # for drug in neg:
    #             #     y_pred.append(drug_scores[drug])
    #
    #             # ===== 使用 neg_index 替代原本的随机负采样 =====
    #             neg_mask = neg_index[0] == ncid
    #             neg_candidates = neg_index[1][neg_mask]  # 所有与 ncid 配对的负drug
    #             filtered_neg = list(set(neg_candidates.tolist()) - set(train_test_ass))  # 排除已知正样本
    #             random.shuffle(filtered_neg)
    #             neg = filtered_neg[:len(pos)]  # 与正样本数一致
    #             # =================================================
    #             # # 收集标签和预测
    #             # y_true += [1] * len(pos)
    #             # y_true += [0] * len(pos)
    #             # for drug in pos:
    #             #     y_pred.append(drug_scores[drug])
    #             # for drug in neg:
    #             #     y_pred.append(drug_scores[drug])
    #
    #             final_len = min(len(pos), len(neg))
    #
    #             y_true += [1] * final_len + [0] * final_len
    #
    #             for drug in pos[:final_len]:
    #                 if drug < len(drug_scores):
    #                     y_pred.append(drug_scores[drug])
    #
    #             for drug in neg[:final_len]:
    #                 if drug < len(drug_scores):
    #                     y_pred.append(drug_scores[drug])
    #
    #     return y_true,y_pred,pred_score_all

    @torch.no_grad()
    def testEpoch(self):
        # 筛选测试集中的正负样本边
        data = torch.load("ncRNADrug2.pth", weights_only=False)
        edge_label = data[("ncRNA", "ncDrug", "drug")].edge_label
        edge_label_index = data[("ncRNA", "ncDrug", "drug")].edge_label_index

        pos_mask = edge_label == 1
        neg_mask = edge_label == 0

        pos_edge_index = edge_label_index[:, pos_mask]
        neg_edge_index = edge_label_index[:, neg_mask]

        # 获取嵌入表示
        ncEmbeds, drugEmbeds = self.model.forward_gcn(self.handler.torchBiAdj)

        # 正样本预测
        pos_preds = torch.sum(
            ncEmbeds[pos_edge_index[0]] * drugEmbeds[pos_edge_index[1]],
            dim=1
        ).cpu().numpy()

        # 负样本预测
        neg_preds = torch.sum(
            ncEmbeds[neg_edge_index[0]] * drugEmbeds[neg_edge_index[1]],
            dim=1
        ).cpu().numpy()

        # 拼接结果
        y_pred = np.concatenate([pos_preds, neg_preds])
        y_true = np.concatenate([
            np.ones(pos_preds.shape[0]),
            np.zeros(neg_preds.shape[0])
        ])

        # 返回完整预测矩阵也可保留
        pred_score_all = torch.mm(ncEmbeds, drugEmbeds.t()).detach().cpu().numpy()

        # 合并正负边索引
        all_edges = torch.cat([pos_edge_index, neg_edge_index], dim=1)  # shape: [2, num_edges]
        all_labels = torch.cat([torch.ones(pos_edge_index.size(1)), torch.zeros(neg_edge_index.size(1))], dim=0)

        # 转为 numpy
        src_nodes = all_edges[0].cpu().numpy()
        dst_nodes = all_edges[1].cpu().numpy()
        #scores = y_pred
        scores_normalized = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
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
        df.to_csv('lncRNA_results.csv', index=False)

        return y_true.tolist(), y_pred.tolist(), pred_score_all.tolist()

    def generator_generate(self, generator):
        """生成增强视图"""
        edge_index = []
        edge_index.append([])
        edge_index.append([])
        adj = deepcopy(self.handler.torchBiAdj)
        idxs = adj._indices()

        with torch.no_grad():
            view = generator.generate(self.handler.torchBiAdj, idxs, adj)

        return view

def seed_it(seed):
    """设置随机种子"""
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)

if __name__ == '__main__':
    with torch.cuda.device(args.gpu):  # 指定GPU
        logger.saveDefault = True  # 启用日志保存
        seed_it(args.seed)  # 设置随机种子

        log('Start')
        handler = DataHandler()  # 初始化数据处理器 DataHandler 12行
        handler.LoadData()  # 加载数据 DataHandler 56行
        log('Load Data')

        coach = Coach(handler)  # 创建训练控制器 Main 18行
        coach.run()  # 启动训练流程