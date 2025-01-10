from torch import nn


class LossModule(nn.Module):
    def __init__(self):
        super(LossModule, self).__init__()
        self.alpha = 0.11

    def forward(self, one_index, zero_index, input, target):
        loss = nn.MSELoss(reduction='none')
        loss_sum = loss(input, target)

        return (1 - self.alpha) * loss_sum[one_index].sum() + self.alpha * loss_sum[zero_index].sum()