# 导入必要的库和模块
from torch import nn  # PyTorch神经网络基础模块
import torch.nn.functional as F  # PyTorch函数式接口
import torch  # PyTorch主库
from Params import args  # 从参数文件导入配置参数
from copy import deepcopy  # 深度拷贝工具
import numpy as np  # 数值计算库
import math  # 数学函数库
import scipy.sparse as sp  # 稀疏矩阵处理
from Utils.Utils import contrastLoss, calcRegLoss, pairPredict  # 自定义工具函数
import time  # 时间相关函数
import torch_sparse  # 稀疏矩阵运算扩展库

# 启用CUDA加速配置
torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True

# 初始化方法定义（Xavier均匀初始化）
init = nn.init.xavier_uniform_


# 主模型类：自适应图对比学习网络，双视图对比学习模块
class AGCLNDA(nn.Module):
	def __init__(self):
		super(AGCLNDA, self).__init__()
		# 用户和项目的嵌入层
		self.uEmbeds = nn.Parameter(init(torch.empty(args.nc, args.latdim)))  # ncRNA嵌入矩阵（nc x latdim）
		self.iEmbeds = nn.Parameter(init(torch.empty(args.drug, args.latdim)))  # drug嵌入矩阵（drug x latdim）
		# 构建多层GCN
		self.gcnLayers = nn.Sequential(*[GCNLayer() for _ in range(args.gnn_layer)])

	# 标准GCN前向传播
	def forward_gcn(self, adj):
		iniEmbeds = torch.cat([self.uEmbeds, self.iEmbeds], axis=0)  # 拼接ncRNA和drug嵌入
		embedsLst = [iniEmbeds]  # 存储各层嵌入

		# 逐层传播
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1])
			embedsLst.append(embeds)
		mainEmbeds = sum(embedsLst)  # 残差连接：各层嵌入求和

		return mainEmbeds[:args.nc], mainEmbeds[args.nc:]  # 拆分RNA和drug嵌入

	# 图对比学习前向传播
	def forward_graphcl(self, adj):
		# 与forward_gcn类似，但返回完整嵌入
		iniEmbeds = torch.cat([self.uEmbeds, self.iEmbeds], axis=0)
		embedsLst = [iniEmbeds]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1])
			embedsLst.append(embeds)
		return sum(embedsLst)

	# 带生成器的图对比学习
	def forward_graphcl_(self, generator):
		iniEmbeds = torch.cat([self.uEmbeds, self.iEmbeds], axis=0)
		embedsLst = [iniEmbeds]
		count = 0
		# 每层使用生成器动态生成邻接矩阵
		for gcn in self.gcnLayers:
			with torch.no_grad():    #不加入梯度
				adj = generator.generate(x=embedsLst[-1], layer=count)  #根据上一层嵌入生成扰动后的邻接矩阵
			embeds = gcn(adj, embedsLst[-1])
			embedsLst.append(embeds)
			count += 1
		return sum(embedsLst)

	# 对比学习损失计算
	def loss_graphcl(self, x1, x2, ncs, drugs):
		T = args.temp # 温度参数
		# 拆分ncRNA和drug嵌入
		nc_embeddings1, drug_embeddings1 = torch.split(x1, [args.nc, args.drug], dim=0)
		nc_embeddings2, drug_embeddings2 = torch.split(x2, [args.nc, args.drug], dim=0)

		# L2归一化
		nc_embeddings1 = F.normalize(nc_embeddings1, dim=1)
		drug_embeddings1 = F.normalize(drug_embeddings1, dim=1)
		nc_embeddings2 = F.normalize(nc_embeddings2, dim=1)
		drug_embeddings2 = F.normalize(drug_embeddings2, dim=1)

		# 获取样本嵌入
		nc_embs1 = F.embedding(ncs, nc_embeddings1)
		drug_embs1 = F.embedding(drugs, drug_embeddings1)
		nc_embs2 = F.embedding(ncs, nc_embeddings2)
		drug_embs2 = F.embedding(drugs, drug_embeddings2)

		# 拼接所有样本
		all_embs1 = torch.cat([nc_embs1, drug_embs1], dim=0)
		all_embs2 = torch.cat([nc_embs2, drug_embs2], dim=0)

		all_embs1_abs = all_embs1.norm(dim=1)
		all_embs2_abs = all_embs2.norm(dim=1)

		# 计算余弦相似度矩阵
		sim_matrix = torch.einsum('ik,jk->ij', all_embs1, all_embs2) / torch.einsum('i,j->ij', all_embs1_abs, all_embs2_abs)
		sim_matrix = torch.exp(sim_matrix / T)# 温度缩放
		# 对比损失计算
		pos_sim = sim_matrix[np.arange(all_embs1.shape[0]), np.arange(all_embs1.shape[0])]
		loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
		loss = - torch.log(loss)

		return loss

	# 获取当前嵌入
	def getEmbeds(self):
		self.unfreeze(self.gcnLayers)
		return torch.cat([self.uEmbeds, self.iEmbeds], axis=0)

	# 解冻指定层的参数
	def unfreeze(self, layer):
		for child in layer.children():
			for param in child.parameters():
				param.requires_grad = True

	# 获取GCN层
	def getGCN(self):
		return self.gcnLayers
# GCN层实现
class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()

	# 前向传播（支持普通spmm和torch_sparse两种模式）
	def forward(self, adj, embeds, flag=True):
		if (flag):
			return torch.spmm(adj, embeds)
		else:
			return torch_sparse.spmm(adj.indices(), adj.values(), adj.shape[0], adj.shape[1], embeds)
# 变分图自编码器编码器
class vgae_encoder(AGCLNDA):
	def __init__(self):
		super(vgae_encoder, self).__init__()
		hidden = args.latdim  # 维度
		# 均值网络
		self.encoder_mean = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden))  # 多层感知器
		# 方差网络
		self.encoder_std = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden), nn.Softplus())  # 多层感知器

	# 重参数化采样
	def forward(self, adj):
		x = self.forward_graphcl(adj)  # 获取基础嵌入
		x_mean = self.encoder_mean(x)  # 均值
		x_std = self.encoder_std(x)  # 标准差
		# 重参数化技巧
		gaussian_noise = torch.randn(x_mean.shape).cuda()
		x = gaussian_noise * x_std + x_mean
		return x, x_mean, x_std

# 变分图自编码器解码器
class vgae_decoder(nn.Module):
	def __init__(self, hidden=args.latdim):
		super(vgae_decoder, self).__init__()
		# 解码器网络
		self.decoder = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1))
		self.sigmoid = nn.Sigmoid()
		self.bceloss = nn.BCEWithLogitsLoss(reduction='none')  #二元交叉熵损失函数

	# x：重参数化后的嵌入（从编码器输出）
	# x_mean, x_std：高斯分布的均值和标准差（用于 KL 散度）
	# ncs, drugs：正样本的 ncRNA 和 drug 索引
	# neg_drugs：负样本的 drug 索引
	# encoder：编码器网络，用于提取参数做正则化
	# 损失计算
	def forward(self, x, x_mean, x_std, ncs, drugs, neg_drugs, encoder):
		x_nc, x_drug = torch.split(x, [args.nc, args.drug], dim=0)
		# 正样本预测
		edge_pos_pred = self.sigmoid(self.decoder(x_nc[ncs] * x_drug[drugs]))
		# 负样本预测
		edge_neg_pred = self.sigmoid(self.decoder(x_nc[ncs] * x_drug[neg_drugs]))
		# 重建损失
		loss_edge_pos = self.bceloss( edge_pos_pred, torch.ones(edge_pos_pred.shape).cuda() )
		loss_edge_neg = self.bceloss( edge_neg_pred, torch.zeros(edge_neg_pred.shape).cuda() )
		loss_rec = loss_edge_pos + loss_edge_neg
		# KL散度
		kl_divergence = - 0.5 * (1 + 2 * torch.log(x_std) - x_mean**2 - x_std**2).sum(dim=1)
		# BPR损失
		ancEmbeds = x_nc[ncs]
		posEmbeds = x_drug[drugs]
		negEmbeds = x_drug[neg_drugs]
		scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
		bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch
		# 正则化
		regLoss = calcRegLoss(encoder) * args.reg
		# 总损失
		beta = 0.1
		loss = (loss_rec + beta * kl_divergence.mean() + bprLoss + regLoss).mean()
		
		return loss
# 完整VGAE模型
class vgae(nn.Module):
	def __init__(self, encoder, decoder):
		super(vgae, self).__init__()
		self.encoder = encoder  # 编码器组件
		self.decoder = decoder  # 解码器组件

	# 前向传播
	def forward(self, data, ncs, drugs, neg_drugs):
		x, x_mean, x_std = self.encoder(data)
		loss = self.decoder(x, x_mean, x_std, ncs, drugs, neg_drugs, self.encoder)
		return loss

	# 生成稀疏邻接矩阵
	def generate(self, data, edge_index, adj):
		x, _, _ = self.encoder(data)
		# 边预测
		edge_pred = self.decoder.sigmoid(self.decoder.decoder(x[edge_index[0]] * x[edge_index[1]]))
		# 生成掩码
		vals = adj._values()
		idxs = adj._indices()
		edgeNum = vals.size()
		edge_pred = edge_pred[:, 0]
		mask = ((edge_pred + 0.5).floor()).type(torch.bool)
		# 构建新邻接矩阵
		newVals = vals[mask]

		newVals = newVals / (newVals.shape[0] / edgeNum[0])
		newIdxs = idxs[:, mask]
		
		return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)
# 去噪网络
class DenoisingNet(nn.Module):
	def __init__(self, gcnLayers, features):
		super(DenoisingNet, self).__init__()

		self.features = features# 初始特征

		self.gcnLayers = gcnLayers# GCN层列表

		self.edge_weights = []
		self.nblayers = []
		self.selflayers = []

		self.attentions = []
		self.attentions.append([])
		self.attentions.append([])

		hidden = args.latdim
		# 自特征提取层
		self.nblayers_0 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
		self.nblayers_1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
		self.nblayers_2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
		self.nblayers_3 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
		self.nblayers_4 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))

		# 自节点特征提取层
		self.selflayers_0 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
		self.selflayers_1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
		self.selflayers_2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
		self.selflayers_3 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
		self.selflayers_4 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))

		# 注意力计算层
		self.attentions_0 = nn.Sequential(nn.Linear( 2 * hidden, 1))
		self.attentions_1 = nn.Sequential(nn.Linear( 2 * hidden, 1))
		self.attentions_2 = nn.Sequential(nn.Linear( 2 * hidden, 1))
		self.attentions_3 = nn.Sequential(nn.Linear( 2 * hidden, 1))
		self.attentions_4 = nn.Sequential(nn.Linear( 2 * hidden, 1))

	# 冻结层参数
	def freeze(self, layer):
		for child in layer.children():
			for param in child.parameters():
				param.requires_grad = False

	# 注意力计算
	def get_attention(self, input1, input2, layer=0):
		if layer == 0:
			nb_layer = self.nblayers_0
			selflayer = self.selflayers_0
		if layer == 1:
			nb_layer = self.nblayers_1
			selflayer = self.selflayers_1
		if layer == 2:
			nb_layer = self.nblayers_2
			selflayer = self.selflayers_2
		if layer == 3:
			nb_layer = self.nblayers_3
			selflayer = self.selflayers_3
		if layer == 4:
			nb_layer = self.nblayers_4
			selflayer = self.selflayers_4

		input1 = nb_layer(input1) #邻居节点特征
		input2 = selflayer(input2) #中心节点特征

		input10 = torch.concat([input1, input2], axis=1)

		if layer == 0:
			weight10 = self.attentions_0(input10)
		if layer == 1:
			weight10 = self.attentions_1(input10)
		if layer == 2:
			weight10 = self.attentions_2(input10)
		if layer == 3:
			weight10 = self.attentions_3(input10)
		if layer == 4:
			weight10 = self.attentions_4(input10)
		
		return weight10

	# Hard Concrete采样
	def hard_concrete_sample(self, log_alpha, beta=1.0, training=True):
		gamma = args.gamma
		zeta = args.zeta

		if training:
			debug_var = 1e-7
			bias = 0.0
			np_random = np.random.uniform(low=debug_var, high=1.0-debug_var, size=np.shape(log_alpha.cpu().detach().numpy()))
			random_noise = bias + torch.tensor(np_random)
			gate_inputs = torch.log(random_noise) - torch.log(1.0 - random_noise)
			gate_inputs = (gate_inputs.cuda() + log_alpha) / beta
			gate_inputs = torch.sigmoid(gate_inputs)
		else:
			gate_inputs = torch.sigmoid(log_alpha)

		stretched_values = gate_inputs * (zeta-gamma) +gamma
		cliped = torch.clamp(stretched_values, 0.0, 1.0)
		return cliped.float()  #返回一个形状与 log_alpha 一致、值在 [0, 1] 的浮点掩码矩阵，用于控制图中哪些边保留、哪些被“关闭”

	def generate(self, x, layer=0):
		f1_features = x[self.row, :]
		f2_features = x[self.col, :]

		weight = self.get_attention(f1_features, f2_features, layer)

		mask = self.hard_concrete_sample(weight, training=False)

		mask = torch.squeeze(mask)
		adj = torch.sparse.FloatTensor(self.adj_mat._indices(), mask, self.adj_mat.shape)

		ind = deepcopy(adj._indices())
		row = ind[0, :]
		col = ind[1, :]

		rowsum = torch.sparse.sum(adj, dim=-1).to_dense()
		d_inv_sqrt = torch.reshape(torch.pow(rowsum, -0.5), [-1])
		d_inv_sqrt = torch.clamp(d_inv_sqrt, 0.0, 10.0)
		row_inv_sqrt = d_inv_sqrt[row]
		col_inv_sqrt = d_inv_sqrt[col]
		values = torch.mul(adj._values(), row_inv_sqrt)
		values = torch.mul(values, col_inv_sqrt)

		support = torch.sparse.FloatTensor(adj._indices(), values, adj.shape)

		return support

	def l0_norm(self, log_alpha, beta):
		gamma = args.gamma
		zeta = args.zeta
		gamma = torch.tensor(gamma)
		zeta = torch.tensor(zeta)
		reg_per_weight = torch.sigmoid(log_alpha - beta * torch.log(-gamma/zeta))

		return torch.mean(reg_per_weight)

	def set_fea_adj(self, nodes, adj):
		self.node_size = nodes
		self.adj_mat = adj

		ind = deepcopy(adj._indices())

		self.row = ind[0, :]
		self.col = ind[1, :]

	def call(self, inputs, training=None):
		if training:
			temperature = inputs
		else:
			temperature = 1.0

		self.maskes = []

		x = self.features.detach()
		layer_index = 0
		embedsLst = [self.features.detach()]

		for layer in self.gcnLayers:
			xs = []
			f1_features = x[self.row, :]
			f2_features = x[self.col, :]

			weight = self.get_attention(f1_features, f2_features, layer=layer_index)
			mask = self.hard_concrete_sample(weight, temperature, training)

			self.edge_weights.append(weight)
			self.maskes.append(mask)
			mask = torch.squeeze(mask)

			adj = torch.sparse.FloatTensor(self.adj_mat._indices(), mask, self.adj_mat.shape).coalesce()
			ind = deepcopy(adj._indices())
			row = ind[0, :]
			col = ind[1, :]

			rowsum = torch.sparse.sum(adj, dim=-1).to_dense() + 1e-6
			d_inv_sqrt = torch.reshape(torch.pow(rowsum, -0.5), [-1])
			d_inv_sqrt = torch.clamp(d_inv_sqrt, 0.0, 10.0)
			row_inv_sqrt = d_inv_sqrt[row]
			col_inv_sqrt = d_inv_sqrt[col]
			values = torch.mul(adj.values(), row_inv_sqrt)
			values = torch.mul(values, col_inv_sqrt)
			support = torch.sparse.FloatTensor(adj._indices(), values, adj.shape).coalesce()

			nextx = layer(support, x, False)
			xs.append(nextx)
			x = xs[0]
			embedsLst.append(x)
			layer_index += 1
		return sum(embedsLst)
	
	def lossl0(self, temperature):
		l0_loss = torch.zeros([]).cuda()
		for weight in self.edge_weights:
			l0_loss += self.l0_norm(weight, temperature)
		self.edge_weights = []
		return l0_loss

	def forward(self, ncs, drugs, neg_drugs, temperature):
		self.freeze(self.gcnLayers)
		x = self.call(temperature, True)
		x_nc, x_drug = torch.split(x, [args.nc, args.drug], dim=0)
		ancEmbeds = x_nc[ncs]
		posEmbeds = x_drug[drugs]
		negEmbeds = x_drug[neg_drugs]
		scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
		bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch
		regLoss = calcRegLoss(self) * args.reg

		lossl0 = self.lossl0(temperature) * args.lambda0
		return bprLoss + regLoss + lossl0




		




