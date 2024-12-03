import pandas as pd
import numpy as np

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def self_similarity_matrix(adj_matrix):
    num_objects = adj_matrix.shape[0]
    similarity_matrix = np.zeros((num_objects, num_objects))
    for i in range(num_objects):
        for j in range(num_objects):
            similarity_matrix[i, j] = cosine_similarity(adj_matrix[i], adj_matrix[j])
    return similarity_matrix


herb_efficacy_df = pd.read_csv("data/herb_efficacy_adj.csv", index_col=0)
target_pathway_df = pd.read_csv("data/target_pathway_adj.csv", index_col=0)

herb_efficacy_adj = herb_efficacy_df.to_numpy()
target_pathway_adj = target_pathway_df.to_numpy()

herb_sim = self_similarity_matrix(herb_efficacy_adj)
target_sim = self_similarity_matrix(target_pathway_adj)
