import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, matthews_corrcoef
)

# 1️⃣ 读取 CSV
df = pd.read_csv("ncRNA_drug_predictions.csv")

# 2️⃣ 根据 score > 0.5 判定预测标签
y_true = df['label'].values
y_score = df['score'].values
y_pred = (y_score > 0.5).astype(int)

# 3️⃣ 计算指标
auc = roc_auc_score(y_true, y_score)                   # AUC
aupr = average_precision_score(y_true, y_score)       # AUPR
acc = accuracy_score(y_true, y_pred)                  # ACC
precision = precision_score(y_true, y_pred)          # 精确率
recall = recall_score(y_true, y_pred)                 # 召回率
f1 = f1_score(y_true, y_pred)                         # F1-score
mcc = matthews_corrcoef(y_true, y_pred)               # MCC

# 4️⃣ 输出结果
print(f"AUC   : {auc:.4f}")
print(f"AUPR  : {aupr:.4f}")
print(f"ACC   : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"MCC     : {mcc:.4f}")
