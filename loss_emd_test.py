import torch
from emd_layer import EMDLoss

# 假设我们设置 n_parts 为 4，即每个 "jet" 有 4 个粒子
n_parts = 4

# 创建 EMDLoss 实例
emd_loss = EMDLoss(n_parts=n_parts)

# 生成两个测试张量 jets1 和 jets2
# 每个 "jet" 有 4 个粒子，每个粒子有 3 个特征 [eta, phi, pt]
# 这里我们创建一个包含 2 个 "jets" 的批次，因此批次大小 nbatch 为 2

# 随机生成 eta 和 phi 特征
eta_phi_jets1 = torch.randint(1 , 11 , (n_parts, 2))  # 形状 [4, 2]
eta_phi_jets2 = torch.randint(1 , 11 , (n_parts, 2)) # 形状 [4, 2]
print("eta_phi_jets1" , eta_phi_jets1 ,'\n' , "eta_phi_jets2" , eta_phi_jets2)

# 随机生成 pt 特征，假设 pt 是正值
pt_jets1 = torch.abs(torch.randint(1 , 11 , (n_parts,)))  # 形状 [4]
pt_jets2 = torch.abs(torch.randint(1 , 11 , (n_parts,)))  # 形状 [4]
print("pt_jets1" , pt_jets1 , '\n' , "pt_jets1" , pt_jets1)

# 将 eta, phi, pt 特征合并为 jets1 和 jets2
jets1 = torch.cat((eta_phi_jets1, pt_jets1.unsqueeze(1)), dim=1)  # 形状 [4, 3]
jets2 = torch.cat((eta_phi_jets2, pt_jets2.unsqueeze(1)), dim=1)  # 形状 [4, 3]
print("jets1" , jets1 , '\n' , "jets2" , jets2)

# 将单个 "jet" 堆叠成批次
jets1 = jets1.unsqueeze(0).repeat(2, 1, 1)  # 形状 [2, 4, 3]
jets2 = jets2.unsqueeze(0).repeat(2, 1, 1)  # 形状 [2, 4, 3]
jets1 = jets1.float()
jets2 = jets2.float()

# 现在 jets1 和 jets2 可以作为输入传递给 EMDLoss
emd_distance, flow = emd_loss(jets1, jets2)

print("EMD Distance:", emd_distance)
print(emd_distance.shape)
print("Flow:", flow)