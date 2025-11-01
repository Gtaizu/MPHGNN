# 导入argparse模块，用于处理命令行参数解析
import argparse


# 定义命令行参数解析函数
def ParseArgs():
    # 创建参数解析器对象，设置程序描述
    parser = argparse.ArgumentParser(description='Model Params')

    # 添加学习率参数，默认值1e-3，浮点类型
    parser.add_argument('--lr', default=5e-5, type=float, help='learning rate')
    # 训练批次大小参数，默认4096，整数类型
    parser.add_argument('--batch', default=1024, type=int, help='batch size')
    # 测试批次大小参数，默认256，整数类型
    parser.add_argument('--tstBat', default=256, type=int, help='number of ncs in a testing batch')
    # 权重衰减正则化系数，默认1e-5，浮点类型
    parser.add_argument('--reg', default=1e-5, type=float, help='weight decay regularizer')
    # 训练总轮次参数，默认30轮，整数类型
    parser.add_argument('--epoch', default=50, type=int, help='number of epochs')
    # 嵌入向量维度参数，默认512维，整数类型
    parser.add_argument('--latdim', default=512, type=int, help='embedding size')
    # GNN层数参数，默认3层，整数类型
    parser.add_argument('--gnn_layer', default=3, type=int, help='number of gnn layers')
    # Top-K评估指标参数，默认K=20，整数类型
    parser.add_argument('--topk', default=20, type=int, help='K of top K')
    # 数据集名称参数，默认'mydata1'，字符串类型
    parser.add_argument('--data', default='mydata1', type=str, help='name of dataset')
    # 对比学习损失权重参数，默认0.1，浮点类型
    parser.add_argument('--ssl_reg', default=0.1, type=float, help='weight for contrastive learning')
    # 信息瓶颈正则化权重参数，默认0.1，浮点类型
    parser.add_argument("--ib_reg", type=float, default=0.1, help='weight for information bottleneck')
    # 对比学习温度参数，默认0.5，浮点类型
    parser.add_argument('--temp', default=0.5, type=float, help='temperature in contrastive learning')
    # 训练时测试间隔轮次参数，默认每1轮测试一次，整数类型
    parser.add_argument('--tstEpoch', default=1, type=int, help='number of epoch to test while training')
    # 指定使用的GPU编号参数，默认3号GPU，整数类型
    parser.add_argument('--gpu', default=0, type=int, help='indicates which gpu to use')
    # 拉普拉斯矩阵L0正则化权重参数，默认1e-4，浮点类型
    parser.add_argument('--lambda0', type=float, default=1e-4, help='weight for L0 loss on laplacian matrix.')
    # Hard Concrete分布参数gamma，默认-0.45，浮点类型（用于稀疏正则化）
    parser.add_argument('--gamma', type=float, default=-0.45)
    # Hard Concrete分布参数zeta，默认1.05，浮点类型（用于稀疏正则化）
    parser.add_argument('--zeta', type=float, default=1.05)
    # 温度参数初始值，默认2.0，浮点类型
    parser.add_argument('--init_temperature', type=float, default=2.0)
    # 温度衰减系数，默认0.98，浮点类型
    parser.add_argument('--temperature_decay', type=float, default=0.98)
    # 数值稳定常数参数，默认1e-3，浮点类型（防止除零错误）
    parser.add_argument("--eps", type=float, default=1e-3)
    # 随机种子参数，默认421，整数类型（确保实验可复现）
    parser.add_argument("--seed", type=int, default=421, help="random seed")

    # 解析命令行参数并返回命名空间对象
    return parser.parse_args()


# 实例化参数对象，全局可通过args.参数名访问
args = ParseArgs()