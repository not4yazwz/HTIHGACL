import pandas as pd
import numpy as np


"""
Datasource(paper): 
Heterogeneous network propagation for herb target identification

Additional file 1: – HIT_herb_target.xls. 23,453 herb-target associations between 1016 herbs and 1214 targets were 
collected and integrated from the HIT database. 
Additional file 2: – CHPA_herb_efficacy.xls. 3487 herb-efficacy associations between 742 herbs and 360 efficacy were 
collected from the Chinese pharmacopoeia (CHPA, 2015 edition). 
Additional file 3: – KEGG_protein_pathway.xls. 16,162 protein-pathway associations between 4794 proteins and 244 
pathways were collected from KEGG database. 
"""
herb_efficacy_df = pd.read_csv("data/herb_efficacy_adj.csv", index_col=0)
target_pathway_df = pd.read_csv("data/target_pathway_adj.csv", index_col=0)
adj_df = pd.read_csv("data/adj.csv", index_col=0)

"""
Include: 3487 herb_efficacy, 16162 target_pathway, 23453 herb_target
"""
herb_efficacy = herb_efficacy_df.to_numpy()
target_pathway = target_pathway_df.to_numpy()
adj = adj_df.to_numpy()

"""
Attention: Exist herb_efficacy, target_pathway, herb_target without associations
"""
related_herb = herb_efficacy.sum(axis=1)
related_target = target_pathway.sum(axis=1)
relate_adj = adj.sum(axis=1)
print(related_herb)
print(related_target)
print(relate_adj)