import torch
import torch.nn as nn
import torch.nn.functional as F
from TNet import TNet, feature_transform_regularizer

# 分类任务的PointNet
class PointNetClassifier(nn.Module):
    def __init__(self,num_class=10):
        super().__init__()

        # 输入对齐
        self.input_tnet = TNet(k=3) # 初始变换矩阵

        # 第一阶段 Shared MLP
        self.conv1 = nn.Conv1d(3,64,1)
        self.bn1 = nn.BatchNorm1d(num_features=64)

        self.conv2 = nn.Conv1d(64,64,1)
        self.bn2 = nn.BatchNorm1d(64)

        # 特征对齐
        self.feature_tnet = TNet(k=64)

        # 第二阶段 Shared MLP
        self.conv3 = nn.Conv1d(64,64,1)
        self.bn3 = nn.BatchNorm1d(64)

        self.conv4 = nn.Conv1d(64,128,1)
        self.bn4 = nn.BatchNorm1d(128)

        self.conv5 = nn.Conv1d(128,1024,1)
        self.bn5 = nn.BatchNorm1d(1024)

        # 分类头
        self.fc1 = nn.Linear(1024, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512,256)
        self.bn_fc2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, num_class)

        self.dropout =nn.Dropout(p=0.2)


    def forward(self,x):
        # x : [B,N,3]
        x = x.transpose(1,2) # [B,3,N]
        T3 = self.input_tnet(x) # T3:[B, 3,3]

        x = torch.bmm(
            x.transpose(1,2), # [B, N, 3]
            T3 # [B, 3,3]
        ).transpose(1,2)    # [B,N,3] -> [B,3,N]

        # Shared MLP
        x = F.relu(self.bn1(self.conv1(x))) # [B, 64,N]
        x = F.relu(self.bn2(self.conv2(x))) # [B, 64, N]
        T64 = self.feature_tnet(x) # T64: [B, 64,64]
        x = torch.bmm(
            x.transpose(1,2), # [B, N, 64]
            T64 # [B, 64,64]
        ).transpose(1,2) #  [B,N,64] -> [B, 64,N]

        # Shared MLP
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x))) # [B, 1024,N]

        # Max Pooling
        x = torch.max(x, dim=2).values # [B, 1024]

        # 分类头
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)

        logits = self.fc3(x)
        return logits, T64

if __name__ == "__main__":
    torch.manual_seed(42)

    B = 4
    N = 256
    K = 10

    points = torch.randn(B, N, 3)
    labels = torch.randint(0, K, (B,))

    model = PointNetClassifier(num_class=K)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001
    )

    model.train()

    logits, T64 = model(points)

    print("分类输出:", logits.shape)


    # 分类损失
    loss_cls = F.cross_entropy(logits, labels)

    # 正交正则化损失
    loss_reg = feature_transform_regularizer(T64)

    # 总损失
    loss = loss_cls + 0.001 * loss_reg

    print(f'分类损失：{loss_cls}')
    print(f'正交正则化损失：{loss_reg}')
    print(f'总损失：{loss}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

