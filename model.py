import torch
from torch import nn
from layer import Contrast_learning, View_attention


class Model(nn.Module):
    def __init__(self, herb_num, target_num, input_dimension, hidden_dimension):
        super(Model, self).__init__()
        self.contrast_learning_herb = Contrast_learning(input_dimension, hidden_dimension)
        self.contrast_learning_target = Contrast_learning(input_dimension, hidden_dimension)

    def forward(self, herb_x, target_x, herb_knn, target_knn, herb_kmeans, target_kmeans):
        herb_x1, herb_x2, herb_cl_loss = self.contrast_learning_herb(herb_x, herb_x, herb_knn, herb_kmeans)

        target_x1, target_x2, target_cl_loss = self.contrast_learning_target(target_x, target_x, target_knn,
                                                                             target_kmeans)
        return
