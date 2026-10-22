import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from PointNet import PointNetClassifier
from TNet import feature_transform_regularizer
from modelnet40_dataset import ModelNet40
from torch.utils.data import random_split  # pytoch自带的划分数据集工具
import matplotlib.pyplot as plt
import time
import random
import numpy as np
import argparse


# 固定随机种子
# 1.固定随机种子函数
def set_seed(seed=42):
    """固定随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PointNet on ModelNet40"
    )

    parser.add_argument(
        "--data-root",
        type=str,
        help="ModelNet40 HDF5 数据目录",
        default=r'data/modelnet40_ply_hdf5_2048/modelnet40_ply_hdf5_2048'
    )
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--num-classes", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epoches", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--model-save-path",
        type=str,
        default="runs/best_pointnet.pt"
    )
    parser.add_argument(
        "--quick-run",
        action="store_true",
        help="快速运行模式：使用少量数据检查代码能否正常运行"
    )
    parser.add_argument(
        "--random-sample",
        action="store_true",
        help="训练时随机采样点云"
    )

    return parser.parse_args()


# 实验配置

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 验证过程
@torch.no_grad()
def evaluate(model, val_loader, num_points):
    model.eval()

    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    for batch_x, batch_y in val_loader:
        batch_x = batch_x[:,:num_points,:]

        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        y_pre, T64 = model(batch_x)
        total_correct += (torch.argmax(y_pre, dim=1) == batch_y).sum().item()
        loss_cls = F.cross_entropy(y_pre, batch_y)
        loss_reg = feature_transform_regularizer(T64)
        loss = loss_cls + 0.001 * loss_reg
        total_loss += loss.item() * len(batch_y)
        total_samples += len(batch_y)

    # 返回这一轮验证的loss和准确率
    return total_loss / total_samples, total_correct / total_samples


# 训练过程
def train(args):
    set_seed(args.seed) # 固定随机种子

    # 创建产生训练结果的文件夹
    run_dir = Path(args.model_save_path).parent
    run_dir.mkdir(parents=True, exist_ok=True)


    print(f'本次训练使用的设备：{device}')

    if args.quick_run:
        args.num_points = 256
        args.epoches = 1

    # Dataset实际加载的点数
    if args.random_sample:
        load_num_points  = 2048
    else:
        load_num_points = args.num_points



    # 完整训练集数据
    full_train_dataset = ModelNet40(
        args.data_root, 'train', load_num_points
    )

    # 完整测试集数据
    test_dataset = ModelNet40(
        args.data_root, 'test', load_num_points
    )

    # 从完整训练集中划分出一部分验证集, 10%作为验证集
    train_size = int(len(full_train_dataset) * 0.9)
    val_size = len(full_train_dataset) - train_size  # 用减法，确保训练集和验证集数据总数为 full_train_dataset中总数，以防random_split报错

    # 专门用于划分数据集的随机数生成器,以保证每次实验的划分随机性是固定的
    split_generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(  # 这个工具并不能保证数据类别均衡
        full_train_dataset,
        [train_size, val_size],
        generator=split_generator
    )

    if args.quick_run:
        # 快速模式仅用于测试模型是否跑通
        train_dataset = torch.utils.data.Subset(
            train_dataset, range(64)
        )

        val_dataset = torch.utils.data.Subset(
            val_dataset, range(32)
        )

    # 创建loader
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    model = PointNetClassifier(args.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = -1
    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []

    # 开始训练
    for epoch in range(args.epoches):
        start_time = time.time()
        total_train_loss = 0.0
        total_train_correct = 0
        train_samples = 0

        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # E1: 只在训练阶段随机采样
            if args.random_sample:
                sampled_list = []
                for points in batch_x:
                    # points: [2048, 3]
                    # 随机选择num_points个索引
                    indices = torch.randperm(len(points), device = points.device)[:args.num_points]
                    # 提取对应的点
                    sampled_points = points[indices] # [1024, 3]
                    sampled_list.append(sampled_points)
                # sample_list:B个[1024,3],使用stack将元素组成一个batch
                batch_x = torch.stack(sampled_list, dim=0) # 可以直接写成 batch_x = torch.tensor(sample_list)吗？不行


            # 前向传播
            y_pre, T64 = model(batch_x)
            loss_cls = F.cross_entropy(y_pre, batch_y)  # 分类损失
            loss_reg = feature_transform_regularizer(T64)  # 正交正则化损失
            loss = loss_cls + 0.001 * loss_reg  # 总损失 = 分类损失 + 正则化损失
            total_train_correct += (torch.argmax(y_pre, dim=1) == batch_y).sum().item()
            total_train_loss += loss.item() * len(batch_y)
            train_samples += len(batch_y)
            # 梯度清零
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss_list.append(total_train_loss / train_samples)
        train_acc_list.append(total_train_correct / train_samples)

        # 验证一轮
        val_loss, val_acc = evaluate(model, val_loader,args.num_points)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        elapsed = time.time() - start_time

        print(f'第{epoch + 1}轮训练：')
        print(
            f'train_loss:{train_loss_list[-1]:.5f},val_loss:{val_loss_list[-1]:.5f}; train_acc:{train_acc_list[-1]:.4f},val_acc:{val_acc_list[-1]:.4f}, 本轮耗时：{elapsed:.1f}s')

        # 模型保存代码
        if val_acc_list[-1] > best_acc:
            best_acc = val_acc_list[-1]
            torch.save(model.state_dict(), args.model_save_path)
            print(f'已保存最佳模型，Val Acc: {best_acc:.4f}')

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'train_loss_list': train_loss_list,
            'val_loss_list': val_loss_list,
            'train_acc_list': train_acc_list,
            'val_acc_list': val_acc_list
        }, run_dir / 'last_check_point.pt')
    if args.epoches > 1:
        # 绘图
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, args.epoches + 1), train_loss_list, label='Training Loss', linewidth=2, color='blue')
        plt.plot(range(1, args.epoches + 1), val_loss_list, label='Validation Loss', linewidth=2, color='red')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss Curve', fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(run_dir / 'loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        # 绘制acc曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, args.epoches + 1), train_acc_list, label='Training Acc', linewidth=2, color='blue')
        plt.plot(range(1, args.epoches + 1), val_acc_list, label='Val Acc', linewidth=2, color='red')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Acc', fontsize=12)
        plt.title('Training and Validation Acc Curve', fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.tight_layout()
        plt.savefig(run_dir / 'acc.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 加载验证集准确率最高的模型
    model.load_state_dict(
        torch.load(
            args.model_save_path,
            map_location=device,
            weights_only=True
        )
    )



    # 使用独立测试集评估
    test_loss, test_acc = evaluate(model, test_loader,args.num_points)

    print('训练结束！')
    print(f'最佳验证准确率：{best_acc:.4f}')
    # print(f'最终测试集 Loss：{test_loss:.5f}')
    # print(f'最终测试集 Acc：{test_acc:.4f}')


if __name__ == '__main__':
    args = parse_args()
    train(args)
