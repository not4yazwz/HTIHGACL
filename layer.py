import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Contrast_learning(nn.Module):
    def __init__(self, input_dimension, hidden_dimension, tau: float = 0.5, alpha: float = 0.5):
        super(Contrast_learning, self).__init__()
        self.tau: float = tau
        self.alpha: float = alpha
        self.hgnn1 = HGNN(input_dimension, hidden_dimension)
        self.hgnn2 = HGNN(input_dimension, hidden_dimension)
        self.fully_connect1 = torch.nn.Linear(hidden_dimension, hidden_dimension)
        self.fully_connect2 = torch.nn.Linear(hidden_dimension, hidden_dimension)

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def cosine_sim(self, z1, z2):
        # calculate cosine similarity
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.cosine_sim(z1, z1))  # 1. intra-view similarity
        between_sim = f(self.cosine_sim(z1, z2))  # 2. inter-view similarity
        # InfoNCE loss
        loss = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        return loss.mean()

    def forward(self, x1, x2, adj1, adj2):
        z1 = self.hgcn1(x1, adj1)
        h1 = self.projection(z1)
        z2 = self.hgcn2(x2, adj2)
        h2 = self.projection(z2)
        loss = self.alpha * self.semi_loss(h1, h2) + (1 - self.alpha) * self.semi_loss(h2, h1)
        return z1, z2, loss


class HGNN(nn.Module):
    def __init__(self, in_dim, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgnn = HGNN_conv(in_dim, n_hid)

    def forward(self, x, G):
        x = F.leaky_relu(self.hgnn(x, G), 0.25)
        return x


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x


class View_attention(nn.Module):
    def __init__(self, node_num, hidden_dimension):
        super(View_attention).__init__()
        self.fully_connect1 = nn.Linear(in_features=2, out_features=self.hiddim)
        self.fully_connect2 = nn.Linear(in_features=self.hiddim, out_features=2)
        self.global_avg_pool = nn.AvgPool2d((hidden_dimension, node_num), (1, 1))
        self.activation = nn.Sigmoid()

    def forward(self, x1, x2):
        concat_x = torch.cat((x1, x2), 1).t()
        concat_x = concat_x.view(1, 1 * 2, x1.shape[1], -1)
        z = self.global_avg_pool(concat_x)
        z = z.view(z.size(0), -1)
        f1 = torch.relu_(self.fully_connect1(z))
        f2 = self.activation(self.fully_connect2(f1))
        channel_attention =f2.view(f2.size(0), f2.size(1), 1, 1)
        z_d = torch.relu(channel_attention * concat_x)
        return z_d
