import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        # Shared MLP
        self.conv1 = nn.Conv1d(k, 64,1)
        self.bn1 = nn.BatchNorm1d(num_features=64)

        self.conv2 = nn.Conv1d(64,128,1)
        self.bn2 = nn.BatchNorm1d(num_features=128)

        self.conv3 = nn.Conv1d(128,1024,1)
        self.bn3 = nn.BatchNorm1d(num_features=1024)

        # FC
        self.fc1 = nn.Linear(1024,512)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512,256)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256,k*k)

        # 初始化权重参数
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)


    def forward(self,x):
        #  输入维度：[B, C, N]
        x = F.relu(self.bn1(self.conv1(x))) # [B, 64, N]
        x = F.relu(self.bn2(self.conv2(x))) # [B, 128, N]
        x = F.relu(self.bn3(self.conv3(x))) # [B, 1024,N]

        # MaxPool
        x = torch.max(x,dim=2).values # [B, 1024]
        x = F.relu(self.bn4(self.fc1(x))) # [B, 512]
        x = F.relu(self.bn5(self.fc2(x)))   # [B, 256]
        x = self.fc3(x) # [B, k*k]
        x = x.reshape(x.shape[0],self.k,self.k) # [B, k,k]

        # 创建单位矩阵
        identity = torch.eye(self.k, device=x.device, dtype=x.dtype).unsqueeze(0) # [1, k,k]的单位矩阵
        return x + identity # 返回残差拟合的线性变化矩阵


# 特征变换正交正则化
def feature_transform_regularizer(T):
    # T:[B, 64, 64]
    k = T.shape[1]
    # 创建单位矩阵
    I = torch.eye(k, device=T.device, dtype=T.dtype).unsqueeze(0) # [B, k,k]
    diff = torch.bmm(T,T.transpose(1,2)) - I # [B, k, k]

    # 对矩阵所有元素求平方，在求和，由于存在Batch,取个均值
    loss = (diff **2).sum(dim=(1,2)).mean()
    return loss



