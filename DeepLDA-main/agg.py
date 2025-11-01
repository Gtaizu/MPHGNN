import torch
from torch_geometric.nn import GATConv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = torch.load("MiDrug_data2.pth", weights_only=False)
# 假设 edge_index 是 gene 节点的内部连接（如同构图边）
edge_index = data["gene", "GeGe", "gene"].edge_index.to(device)  # 形状 [2, num_edges]
x1 = data["miRNA-gene"].x.to(device)
x2 = data["gene-drug"].x.to(device)
# 初始化 GAT 层
gat = GATConv(in_channels=400, out_channels=200, heads=1)
gat = gat.to(device)
# 将两个特征矩阵拼接后输入 GAT
combined_x = torch.cat([x1, x2], dim=1)
# 拼接特征并计算（禁用梯度）
with torch.no_grad():
    aggregated_x = gat(combined_x, edge_index)  # 输出 [634, 200]

data1 = torch.load("MiDrug_data.pth", weights_only=False)
data1["gene"].node_id = torch.arange(aggregated_x.size(0))
data1["gene"].x = aggregated_x
print(data1["gene"].x)
print(data1)
torch.save(data1, "MiDrug_data_end.pth")