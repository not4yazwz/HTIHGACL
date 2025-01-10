import torch
from torch import nn
from layer import Contrast_learning, View_attention, TransformerEncoder


class Model(nn.Module):
    def __init__(self, herb_num, target_num, input_dimension, hidden_dimension):
        super(Model, self).__init__()
        self.contrast_learning_herb = Contrast_learning(input_dimension, hidden_dimension)
        self.contrast_learning_target = Contrast_learning(input_dimension, hidden_dimension)
        self.view_attention_herb = View_attention(herb_num, hidden_dimension)
        self.view_attention_target = View_attention(target_num, hidden_dimension)
        self.transformer_herb = TransformerEncoder([hidden_dimension, hidden_dimension])
        self.transformer_target = TransformerEncoder([hidden_dimension, hidden_dimension])

        self.linear_x_1 = nn.Linear(hidden_dimension, 256)
        self.linear_x_2 = nn.Linear(256, 128)
        self.linear_x_3 = nn.Linear(128, 64)

        self.linear_y_1 = nn.Linear(hidden_dimension, 256)
        self.linear_y_2 = nn.Linear(256, 128)
        self.linear_y_3 = nn.Linear(128, 64)


    def forward(self, herb_x, target_x, herb_knn, target_knn, herb_kmeans, target_kmeans):

        # 1.超图 + 对比学习 | shape (node_num, hidden_dim)
        herb_x1, herb_x2, herb_cl_loss = self.contrast_learning_herb(herb_x, herb_x, herb_knn, herb_kmeans)

        # 2.自适应多来源特征权重 | shape (1, 2, node_num, hidden_dim)
        herb_attention_x = self.view_attention_herb(herb_x1, herb_x2)

        # 3.transformer | shape (1, node_num, hidden_dim)
        herb_concat_x = torch.cat([herb_attention_x[0].t(), herb_attention_x[1].t()], dim=1)
        herb_x = self.transformer_herb(herb_concat_x)

        target_x1, target_x2, target_cl_loss = self.contrast_learning_target(target_x, target_x, target_knn, target_kmeans)
        target_attention_x = self.view_attention_target(target_x1, target_x2)
        target_concat_x = torch.cat([target_attention_x[0].t(), target_attention_x[1].t()], dim=1)
        target_x = self.transformer_target(target_concat_x)

        # 4.MLP
        herb_fc1 = torch.relu(self.linear_x_1(herb_x))
        herb_fc2 = torch.relu(self.linear_x_2(herb_fc1))
        herb_fc = torch.relu(self.linear_x_3(herb_fc2))

        target_fc1 = torch.relu(self.linear_y_1(target_x))
        target_fc2 = torch.relu(self.linear_y_2(target_fc1))
        target_fc = torch.relu(self.linear_y_3(target_fc2))

        score = herb_fc.mm(target_fc.t())

        return score, herb_cl_loss, target_cl_loss
