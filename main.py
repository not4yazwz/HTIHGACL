import numpy as np
import torch
from torch.onnx.symbolic_opset9 import tensor
from KNN import  knn
from Kmeans import kmeans

# 1、参数设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
herb_num = 100
target_num = 200
cv = 5

# 2、数据读取
adj = np.zeros([herb_num, target_num])

# 3、相似矩阵构建
herb_sim_efficacy = np.zeros([herb_num, herb_num])
herb_sim_gip = np.zeros([herb_num, herb_num])
herb_sim = np.where(herb_sim_efficacy == 0, herb_sim_gip, (herb_sim_efficacy + herb_sim_gip) / 2)

target_sim_pathway = np.zeros([target_num, target_num])
target_sim_gip = np.zeros([target_num, target_num])
target_sim = np.where(target_sim_pathway == 0, target_sim_gip, (target_sim_pathway + target_sim_gip) / 2)

# 4、取出正负样本并打乱
torch.manual_seed(0)
zero_tensor = torch.LongTensor(np.argwhere(adj == 0).tolist())
one_tensor = torch.LongTensor(np.argwhere(adj == 1).tolist())
zero_tensor = zero_tensor[torch.randperm(zero_tensor.size(0))]
one_tensor = one_tensor[torch.randperm(one_tensor.size(0))]

# 5、划分 10 份，前 9 份做交叉验证，最后 1 份独立实验（最终测试）
split_size_zero = zero_tensor.size(0) // 10
zero_splits = zero_tensor.split(split_size_zero, dim=0)
cross_zero = torch.cat(zero_splits[:9], dim=0)
split_size_one = one_tensor.size(0) // 10
one_splits = one_tensor.split(split_size_one, dim=0)
cross_one = torch.cat(one_splits[:9], dim=0)
new_zero_splits = cross_zero.split(cross_zero.size(0) // cv, dim=0)
new_one_splits = cross_one.split(cross_one.size(0) // cv, dim=0)
new_zero_splits = list(new_zero_splits)
new_one_splits = list(new_one_splits)

data = {
    "herb_sim": torch.from_numpy(herb_sim),
    "target_sim": torch.from_numpy(target_sim),
    "cross_validation": [],
    "final_test": []
}

for i in range(cv):
    test_zero = new_zero_splits[i]
    test_one = new_one_splits[i]
    train_zero = torch.cat([new_zero_splits[j] for j in range(cv) if j != i], dim=0)
    train_one = torch.cat([new_one_splits[j] for j in range(cv) if j != i], dim=0)
    data['cross_validation'].append({
        'test': [test_one, test_zero],
        'train': [train_one, train_zero]
    })

ind_zero_test = zero_splits[-1]
ind_one_test = one_splits[-1]
data['final_test'].append({
    'test': [ind_one_test, ind_zero_test],
    'train': [cross_one, cross_zero]
})

"""==============================================================================================="""
#
herb_sim_tensor = data['herb_sim'].to(device)
target_sim_tensor = data['target_sim'].to(device)
concat_herb = np.hstack([adj, herb_sim])
concat_herb_tensor = tensor.FloatTensor(concat_herb).to(device)
concat_target = np.hstack([adj.T, target_sim])
concat_target_tensor = tensor.FloatTensor(concat_target).to(device)

# K-means and KNN capture features
knn_herb = knn(concat_herb_tensor.detach().cpu().numpy(), 13).to(device)
knn_target = knn(concat_target_tensor.detach().cpu().numpy(), 13).to(device)
kmeans_herb = kmeans(concat_herb_tensor.detach().cpu().numpy(), 9).to(device)
kmeans_target = kmeans(concat_target_tensor.detach().cpu().numpy(), 9).to(device)

"""==============================================================================================="""
