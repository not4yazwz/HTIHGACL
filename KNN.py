"""
使用KNN构建超图关联矩阵H。

参数：
- X: 特征矩阵，形状为 (n_samples, n_features)
- k: 邻居数量

返回：
- H: 超图关联矩阵，形状为 (n_samples, n_samples)

"""
import numpy as np
import torch
from scipy.sparse import diags, csr_matrix
from sklearn.neighbors import NearestNeighbors


def knn(node_feature, neighbor_num):
    n_samples = node_feature.shape[0]
    # 使用KNN找到每个样本的k个近邻（包括自身）
    nearest = NearestNeighbors(n_neighbors=neighbor_num + 1, algorithm='auto').fit(node_feature)
    distances, indices = nearest.kneighbors(node_feature)

    # 初始化超图关联矩阵H
    h = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        # 获取k个近邻的索引，排除自身
        neighbors = indices[i][1:]
        # 为每个样本创建一个超边，连接自身和其k个近邻
        h[neighbors, i] = 1
        h[i, i] = 1

    g = generate_g_from_h(h, variable_weight=True)
    return g


def generate_g_from_h(H, variable_weight=False):
    """
    根据超图关联矩阵 H 生成归一化的超图邻接矩阵 G。

    参数：
    - H: 超图关联矩阵，形状为 (n_samples, n_edges)
    - variable_weight: 是否返回分解后的矩阵

    返回：
    - G: 归一化的超图邻接矩阵（如果 variable_weight=False）
    - 或分解后的矩阵（如果 variable_weight=True）
    """
    H = csr_matrix(H)
    n_samples, n_edges = H.shape
    W = np.ones(n_edges)

    # 计算节点度向量 DV 和超边度向量 DE
    DV = np.array(H.dot(W)).flatten()
    DE = np.array(H.sum(axis=0)).flatten()

    epsilon = 1e-5  # 避免除零
    DV[DV == 0] = epsilon
    DE[DE == 0] = epsilon

    # 构建度的逆矩阵
    invDE = diags(1.0 / DE)
    DV2 = diags(1.0 / np.sqrt(DV))

    HT = H.transpose()

    if variable_weight:
        DV2_H = DV2.dot(H)
        invDE_HT_DV2 = invDE.dot(HT).dot(DV2)
        return DV2_H, W, invDE_HT_DV2
    else:
        W = diags(W)
        G = DV2.dot(H).dot(W).dot(invDE).dot(HT).dot(DV2)
        G = torch.Tensor(G.toarray())
        return G
