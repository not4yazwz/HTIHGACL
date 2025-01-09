import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 读取数据
herb_efficacy_df = pd.read_csv("data/herb_efficacy_adj.csv", index_col=0)
target_pathway_df = pd.read_csv("data/target_pathway_adj.csv", index_col=0)

# 转换为 NumPy 数组
herb_efficacy_adj = herb_efficacy_df.to_numpy()
target_pathway_adj = target_pathway_df.to_numpy()

# 计算自相似矩阵
herb_sim = cosine_similarity(herb_efficacy_adj)
target_sim = cosine_similarity(target_pathway_adj)


