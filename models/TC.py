import numpy as np
import torch
import torch.nn as nn

from .attention import Seq_Transformer

class TC(nn.Module):
    def __init__(self, configs, device):
        super(TC, self).__init__()
        self.num_channels = configs.final_out_channels
        self.timestep = configs.TC.timesteps
        self.Wk = nn.ModuleList([nn.Linear(configs.TC.hidden_dim, self.num_channels) for i in range(self.timestep)])  # ?全连接层
        self.lsoftmax = nn.LogSoftmax()
        self.device = device
        
        self.projection_head = nn.Sequential(
            nn.Linear(configs.TC.hidden_dim, configs.final_out_channels // 2),
            nn.BatchNorm1d(configs.final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(configs.final_out_channels // 2, configs.final_out_channels // 4),
        )
        
        self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=configs.TC.hidden_dim, depth=4, heads=4, mlp_dim=64)
    def forward(self, z_aug1, z_aug2):
        seq_len = z_aug1.shape[2]
        
        z_aug1 = z_aug1.transpose(1, 2)  # 转置，目的是实现交换维度
        z_aug2 = z_aug2.transpose(1, 2)
        
        batch = z_aug1.shape[0]
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(self.device)  # 随机选择时间戳
        
        nce = 0  # 时间步长和批次的平均值
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)  # 构造空矩阵
        
        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z_aug2[:, t_samples + i, :].view(batch, self.num_channels)  # 随着i的增加，取某个时间步上的
        forward_seq = z_aug1[:, :t_samples + 1, :]
        
        c_t = self.seq_transformer(forward_seq)
        
        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # 做预测
        nce /= -1. * batch * self.timestep
        return nce, self.projection_head(c_t)
