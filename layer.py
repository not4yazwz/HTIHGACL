import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Contrast_learning(nn.Module):
    """对比学习"""
    def __init__(self, input_dimension, hidden_dimension, tau: float = 0.5, alpha: float = 0.5):
        super(Contrast_learning, self).__init__()
        self.tau: float = tau
        self.alpha: float = alpha
        self.hgnn1 = HGNN(input_dimension, hidden_dimension)
        self.hgnn2 = HGNN(input_dimension, hidden_dimension)
        self.fc1 = torch.nn.Linear(hidden_dimension, hidden_dimension)
        self.fc2 = torch.nn.Linear(hidden_dimension, hidden_dimension)

    def projection(self, z):
        # 投影层
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def cosine_sim(self, z1, z2):
        # calculate cosine similarity
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1, z2):
        # 半监督对比损失
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.cosine_sim(z1, z1))  # 1. intra-view similarity
        between_sim = f(self.cosine_sim(z1, z2))  # 2. inter-view similarity
        # InfoNCE loss：
        # between_sim的对角线即{vi,ui}
        loss = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        return loss.mean()

    def forward(self, x1, x2, adj1, adj2):
        z1 = self.hgnn1(x1, adj1)
        h1 = self.projection(z1)
        z2 = self.hgnn2(x2, adj2)
        h2 = self.projection(z2)
        loss = self.alpha * self.semi_loss(h1, h2) + (1 - self.alpha) * self.semi_loss(h2, h1)
        return z1, z2, loss


class HGNN(nn.Module):
    """增加一层激活函数"""
    def __init__(self, in_dim, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgnn = HGNN_conv(in_dim, n_hid)

    def forward(self, x, G):
        x = F.leaky_relu(self.hgnn(x, G), 0.25)
        return x


class HGNN_conv(nn.Module):
    """单独一个Hyper Graph Neural Network卷积层"""
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        # 可训练权重
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        # 可训练偏置
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化参数，使用 uniform (均匀分布)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        # 输入的 G 为归一化的超图邻接矩阵
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

    def __repr__(self):
        # print(layer) 会输出 HGNN_conv (in_features -> out_features)
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


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
        f1 = torch.relu(self.fully_connect1(z))
        f2 = self.activation(self.fully_connect2(f1))
        channel_attention = f2.view(f2.size(0), f2.size(1), 1, 1)
        z_d = torch.relu(channel_attention * concat_x)
        return z_d


class Attention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        # softmax + dropout
        attn = attn / abs(attn.min())
        attn = self.dropout(F.softmax(F.normalize(attn, dim=-1), dim=-1))
        output = torch.matmul(attn, v)

        return output, attn, v


class EncodeLayer(nn.Module):
    # Transformer编码器
    def __init__(self, d_model, d_k, d_v, n_head, dropout):
        super(EncodeLayer, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_head * d_v, bias=False)

        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = Attention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, modal_num, mask=None):
        bs = q.size(0)
        residual = q
        q = self.w_q(q).view(bs, modal_num, self.n_head, self.d_k)
        k = self.w_k(k).view(bs, modal_num, self.n_head, self.d_k)
        v = self.w_v(v).view(bs, modal_num, self.n_head, self.d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, attn, _ = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(bs, modal_num, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        return q, attn


class VariLengthInputLayer(nn.Module):
    def __init__(self, input_data_dims, d_k, d_v, n_head, dropout):
        super(VariLengthInputLayer, self).__init__()
        self.n_head = n_head
        self.dims = input_data_dims
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = []
        self.w_ks = []
        self.w_vs = []
        for i, dim in enumerate(self.dims):
            self.w_q = nn.Linear(dim, n_head * d_k, bias=False)
            self.w_k = nn.Linear(dim, n_head * d_k, bias=False)
            self.w_v = nn.Linear(dim, n_head * d_v, bias=False)

            self.w_qs.append(self.w_q)
            self.w_ks.append(self.w_k)
            self.w_vs.append(self.w_v)
            self.add_module('linear_q_%d_%d' % (dim, i), self.w_q)
            self.add_module('linear_k_%d_%d' % (dim, i), self.w_k)
            self.add_module('linear_v_%d_%d' % (dim, i), self.w_v)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.attention = Attention(temperature=d_k ** 0.5, attn_dropout=dropout)
        self.fc = nn.Linear(n_head * d_v, n_head * d_v)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(n_head * d_v, eps=1e-6)

    def forward(self, input_data, mask=None):

        temp_dim = 0
        bs = input_data.size(0)
        modal_num = len(self.dims)
        q = torch.zeros(bs, modal_num, self.n_head * self.d_k).to(self.device)
        k = torch.zeros(bs, modal_num, self.n_head * self.d_k).to(self.evice)
        v = torch.zeros(bs, modal_num, self.n_head * self.d_v).to(self.device)

        for i in range(modal_num):
            w_q = self.w_qs[i]
            w_k = self.w_ks[i]
            w_v = self.w_vs[i]

            data = input_data[:, temp_dim: temp_dim + self.dims[i]]
            temp_dim += self.dims[i]
            q[:, i, :] = w_q(data)
            k[:, i, :] = w_k(data)
            v[:, i, :] = w_v(data)

        q = q.view(bs, modal_num, self.n_head, self.d_k)
        k = k.view(bs, modal_num, self.n_head, self.d_k)
        v = v.view(bs, modal_num, self.n_head, self.d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, attn, residual = self.attention(q, k, v)
        q = q.transpose(1, 2).contiguous().view(bs, modal_num, -1)
        residual = residual.transpose(1, 2).contiguous().view(bs, modal_num, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)

        return q, attn


class FeedForwardLayer(nn.Module):
    # 前馈传播层：两层线性层 + 残差 + 正则化
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.gelu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, input_data_dims, hyperpm):
        super(TransformerEncoder, self).__init__()
        self.hyperpm = hyperpm
        self.input_data_dims = input_data_dims
        self.d_q = hyperpm.n_hidden
        self.d_k = hyperpm.n_hidden
        self.d_v = hyperpm.n_hidden
        self.n_head = hyperpm.n_head
        self.dropout = hyperpm.dropout
        self.n_layer = hyperpm.nlayer
        self.modal_num = hyperpm.nmodal
        self.d_out = self.d_v * self.n_head * self.modal_num

        self.InputLayer = VariLengthInputLayer(self.input_data_dims, self.d_k, self.d_v, self.n_head, self.dropout)

        self.Encoder = []
        self.FeedForward = []

        for i in range(self.n_layer):
            encoder = EncodeLayer(self.d_k * self.n_head, self.d_k, self.d_v, self.n_head, self.dropout)
            self.add_module('encode_%d' % i, encoder)
            self.Encoder.append(encoder)

            feedforward = FeedForwardLayer(self.d_v * self.n_head, self.d_v * self.n_head, dropout=self.dropout)
            self.add_module('feed_%d' % i, feedforward)
            self.FeedForward.append(feedforward)


    def forward(self, x):
        bs = x.size(0)
        attn_map = []
        x, _attn = self.InputLayer(x)

        attn = _attn.mean(dim=1)
        attn_map.append(attn.detach().cpu().numpy())

        for i in range(self.n_layer):
            x, _attn = self.Encoder[i](q=x, k=x, v=x, modal_num=self.modal_num)
            attn = _attn.mean(dim=1)
            x = self.FeedForward[i](x)
            attn_map.append(attn.detach().cpu().numpy())

        x = x.view(bs, -1)

        # output = self.Outputlayer(x)
        return x