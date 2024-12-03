"""
输入:
1、node_feature 大小为 N x d, Tensor格式
2、cluster_num 族数量

输出:
incidence matrix 大小为 N x cluster_num

处理示例:
A 1 3 4 5
B 2 1 2 4
C 2 1 1 4
D 4 4 1 2
input features: 4 nodes, 4 dimensions
cluster_num: 2
------------>
labels_: [0, 1, 1, 0]
A 1 0
B 0 1
C 0 1
D 1 0
output: 4 nodes, 2 dimensions

"""
import numpy as np
import torch
from sklearn.cluster import KMeans


def kmeans(node_feature, cluster_num):
    model = KMeans(n_clusters=cluster_num, init='k-means++', random_state=0)
    labels = model.fit_predict(node_feature)
    incidence = np.eye(cluster_num)[labels]

    # 初始化权重矩阵 w: cluster x cluster
    w = np.eye(incidence.shape[1])

    # 计算节点度矩阵 dv 的 -1/2 次方
    dv = np.sum(incidence, axis=1)  # 节点的度
    dv_inv_sqrt = np.diag(np.power(dv, -0.5, where=(dv != 0)))

    # 计算超边度矩阵 de 的 -1 次方
    de = np.sum(incidence, axis=0)  # 超边的度
    de_inv = np.diag(np.power(de, -1, where=(de != 0)))

    # 将关联矩阵转换为 PyTorch 张量
    h = torch.tensor(incidence, dtype=torch.float32)
    dv_inv_sqrt = torch.tensor(dv_inv_sqrt, dtype=torch.float32)
    de_inv = torch.tensor(de_inv, dtype=torch.float32)
    w = torch.tensor(w, dtype=torch.float32)

    # 计算邻接矩阵 g
    g = dv_inv_sqrt @ h @ w @ de_inv @ h.T @ dv_inv_sqrt

    return g

