# 导入必要的库
import pickle
import random

import numpy as np
import torch
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from Params import args  # 导入参数配置
import scipy.sparse as sp
from Utils.TimeLogger import log  # 日志工具
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader

class DataHandler:
    def __init__(self):
        # 初始化数据路径
        predir = './Datasets/mydata1/'  # 数据集目录
        self.predir = predir
        self.trnfile = predir + 'train0'  # 训练数据文件路径
        self.tstfile = predir + 'test0'   # 测试数据文件路径

    def loadOneFile(self, filename):
        # 加载单个文件并转换为稀疏矩阵
        with open(filename, 'rb') as fs:
            # 读取pickle文件，非零元素转换为1.0，类型为float32
            ret = (pickle.load(fs) != 0).astype(np.float32)
        # 确保返回的是coo_matrix格式
        if type(ret) != coo_matrix:
            ret = sp.coo_matrix(ret)
        return ret

    def normalizeAdj(self, mat):
        # 对称归一化邻接矩阵
        degree = np.array(mat.sum(axis=-1))  # 计算度矩阵
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])  # 度矩阵的-1/2次方
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0  # 处理无穷大值
        dInvSqrtMat = sp.diags(dInvSqrt)  # 转换为对角矩阵
        # 归一化：D^(-1/2) * A * D^(-1/2)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def makeTorchAdj(self, mat):
        # 构建PyTorch格式的邻接矩阵
        # 创建用户和项目的零矩阵占位符
        a = sp.csr_matrix((args.nc, args.nc))  # ncRNA-ncRNA初始零矩阵
        b = sp.csr_matrix((args.drug, args.drug))  # drug-drug初始零矩阵
        # 拼接成块矩阵：[[A, R], [R^T, B]]，其中R是交互矩阵
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0  # 二值化，非零元素置1
        mat = (mat + sp.eye(mat.shape[0])) * 1.0  # 添加自连接
        mat = self.normalizeAdj(mat)  # 归一化

        # 转换为PyTorch稀疏张量并移到GPU
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))  # 非零元素的行列索引
        vals = t.from_numpy(mat.data.astype(np.float32))  # 非零元素的值
        shape = t.Size(mat.shape)  # 矩阵形状
        return t.sparse.FloatTensor(idxs, vals, shape).cuda()  # 创建稀疏张量并移至CUDA

    def LoadData(self):
        # 加载并处理数据
        # trnMat = self.loadOneFile(self.trnfile)  # 加载训练数据
        # tstMat = self.loadOneFile(self.tstfile)   # 加载测试数据

        data = torch.load("ncRNADrug2.pth", weights_only=False)
        # 提取 edge_label_index 和 edge_label
        edge_label_index = data[("ncRNA", "ncDrug", "drug")].edge_label_index
        edge_label = data[("ncRNA", "ncDrug", "drug")].edge_label
        edge_index = data[("ncRNA", "ncDrug", "drug")].edge_index
        # 找到标签为 1 的边的索引位置
        pos_indices = (edge_label == 1).nonzero(as_tuple=True)[0]

        # 提取对应的边（正样本边）
        test_pos_edge_index = edge_label_index[:, pos_indices]

        # 构建训练集和测试集的 pos_edge_index
        train_pos_edge_index = edge_index
        test_pos_edge_index = test_pos_edge_index

        row = train_pos_edge_index[0].cpu().numpy()
        col = train_pos_edge_index[1].cpu().numpy()
        data1 = np.ones_like(row, dtype=np.float32)

        #trnMat = sp.coo_matrix((data1, (row, col)), shape=(max(new_lnc_index) + 1, max(new_drug_index) + 1),dtype=np.float32)
        trnMat = sp.coo_matrix((data1, (row, col)), shape=(data['ncRNA'].x.size(0), data['drug'].x.size(0)), dtype=np.float32)
        print(trnMat.shape)

        row = test_pos_edge_index[0].cpu().numpy()
        col = test_pos_edge_index[1].cpu().numpy()
        data1 = np.ones_like(row, dtype=np.float32)

        #tstMat = sp.coo_matrix((data1, (row, col)), shape=(max(new_lnc_index) + 1, max(new_drug_index) + 1),dtype=np.float32)
        tstMat = sp.coo_matrix((data1, (row, col)), shape=(data['ncRNA'].x.size(0), data['drug'].x.size(0)), dtype=np.float32)
        print(tstMat.shape)

        self.trnMat = trnMat
        args.nc, args.drug = trnMat.shape  # 设置ncRNA数nc和drug数drug
        self.torchBiAdj = self.makeTorchAdj(trnMat)  # 生成邻接矩阵张量

        # 训练数据加载器
        trnData = TrnData(trnMat)
        self.trnData = trnData
        self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)

        # 测试数据加载器
        tstData = TstData(tstMat, trnMat)
        self.tstData = tstData
        self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)

# 构造训练集样本
class TrnData(data.Dataset):
    def __init__(self, coomat):
        # 初始化训练数据集
        # 分别保存交互对的 ncRNA 和 drug 索引（非零项），用于训练的正样本
        self.rows = coomat.row
        self.cols = coomat.col
        # 将稀疏 COO 矩阵转为 DOK（Dictionary Of Keys）格式，便于快速判断某个 (nc, drug) 是否存在
        self.dokmat = coomat.todok()
        # 创建一个空的负样本数组 self.negs，长度等于正样本数量
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

        # 记录每个RNA与drug交互过（正样本）
        trnLocs = [None] * coomat.shape[0]  # 第 i 个 ncRNA 的正交互 drug ID 列表
        trnncs = set()  # 有交互的ncRNA集合
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            if trnLocs[row] is None:
                trnLocs[row] = list()
            trnLocs[row].append(col)
            trnncs.add(row)
        self.trnncs = np.array(list(trnncs))  # 转换为数组
        self.trnLocs = trnLocs  # 每个RNA的训练项目列表

    def negSampling(self):
        # 负采样：为每个正样本随机选取一个未交互的项目
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                iNeg = np.random.randint(args.drug)  # 随机采样
                if (u, iNeg) not in self.dokmat:  # 确保未在训练集中出现
                    break
            self.negs[i] = iNeg

    def __len__(self):
        # 返回数据集大小（正样本数量）
        return len(self.rows)

    def __getitem__(self, idx):
        # 获取一个训练样本：用户、正项目、负项目
        return self.rows[idx], self.cols[idx], self.negs[idx]

class TstData(data.Dataset):
    def __init__(self, coomat, trnMat):
        # 初始化测试数据集
        self.csrmat = (trnMat.tocsr() != 0) * 1.0  # 训练矩阵二值化并转为CSR格式

        # 记录每个用户的测试项目
        tstLocs = [None] * coomat.shape[0]
        tstncs = set()  # 有测试交互的用户集合
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            if tstLocs[row] is None:
                tstLocs[row] = list()
            tstLocs[row].append(col)
            tstncs.add(row)
        self.tstncs = np.array(list(tstncs))  # 转换为数组
        self.tstLocs = tstLocs  # 每个用户的测试项目列表

    def __len__(self):
        # 返回测试ncRNA数量
        return len(self.tstncs)

    def __getitem__(self, idx):
        # 获取一个测试ncRNA及其历史交互向量
        return self.tstncs[idx], np.reshape(self.csrmat[self.tstncs[idx]].toarray(), [-1])