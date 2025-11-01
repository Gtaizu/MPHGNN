import torch
import argparse
from utils import get_data, MaskPath, print_result, set_seed
from model import GNNEncoder, EdgeDecoder, GAM
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合服务器环境
import matplotlib.pyplot as plt

# main parameter
# 创建一个参数解析器对象，用于从命令行接收参数
parser = argparse.ArgumentParser()
parser.add_argument('--layer', default="gcn", help="sage, gcn, gin, gat, gat2")
parser.add_argument('--seed', type=int, default=2023, help="Random seed for model and dataset.")
# num_encoder: GNNEncoder 中 GNN 层的数量（比如 2 层 GCN）
# num_decoder: EdgeDecoder 中 MLP 层的数量
parser.add_argument('--num_encoder', type=int, default=2, help="numbers of GNN encoder")
parser.add_argument('--num_decoder', type=int, default=2, help="numbers of Edge decoder")
# walk_length: 随机游走的步长（遮蔽边路径长度）
# p: 遮蔽起点选择概率（决定有多少节点开始游走）
parser.add_argument('--walk_length', type=int, default=3, help="length of walk")
parser.add_argument('--p', type=float, default=0.3, help='Mask ratio')
# lr: 学习率，用于 Adam 优化器
# wd: 权重衰减（正则化项）
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate in optimizer')
parser.add_argument('--wd', type=float, default=5e-4, help='weight decay in optimizer')
# times: 重复训练实验的次数（用于多次试验评估模型稳定性）
# epoch: 每次训练的迭代轮数
parser.add_argument('--times', type=int, default=10, help="numbers of training times")
parser.add_argument('--epoch', type=int, default=30, help="numbers of training epoch")
# 从命令行读取参数
args = parser.parse_args()

set_seed(args.seed)

data = get_data()  # utils.py 101行
mask = MaskPath(p=args.p, num_nodes=len(data['train'].x), walk_length=args.walk_length)  # utils 73行
encoder = GNNEncoder(len(data['train'].x[0]), 128, 256, num_layers=args.num_encoder, layer=args.layer)  # model 23行
edge_decoder = EdgeDecoder(256, 64, 1, num_layers=args.num_decoder)  # model 65行

model = GAM(encoder, edge_decoder, mask).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

all_result = []
for x in range(args.times):
    for epoch in range(args.epoch):
        # 设置为训练模式
        model.train()
        loss = model.train_epoch(data, optimizer)
    # 设置为评估模式
    model.eval()
    test_data = data['test']
    z = model.encoder(test_data.x, test_data.edge_index)
    result = model.test(z, test_data.pos_edge_label_index, test_data.neg_edge_label_index, args.layer)  # model 154行
    all_result.append(result)
print_result(all_result)
# plt.show()
