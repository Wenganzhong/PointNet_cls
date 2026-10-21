# PointNet Classification Baseline

基于 PyTorch 复现 PointNet，并在 ModelNet40 数据集上完成三维点云分类。本仓库保存可复现的 baseline，后续将在相同数据划分和评估流程上进行数据增强、训练策略及网络结构实验。

## Baseline 结果

| 指标 | 结果 |
| --- | ---: |
| 最佳验证集准确率 | 87.50% |
| 测试集准确率 | 86.51% |
| 测试集 Loss | 0.45917 |
| 输入点数 | 1024 |
| 训练轮数 | 30 |
| Batch size | 16 |
| 学习率 | 0.001 |
| 随机种子 | 2026 |

训练集和验证集由官方训练集按 9:1 随机划分，官方测试集仅用于最终评估。当前数据副本的样本指纹检查未发现训练集、验证集和测试集之间存在完全重复点云。

## 项目结构

```text
PointNet_cls/
├── PointNet.py               # PointNet 分类网络
├── TNet.py                   # 输入与特征变换网络
├── modelnet40_dataset.py     # ModelNet40 数据读取
├── datatest.py               # 数据集读取测试
├── train_pointnet.py         # 训练、验证和测试入口
├── data/                     # 数据集目录（Git 忽略）
└── runs/                     # 权重、日志和曲线（Git 忽略）
```

## 环境依赖

- Python 3.10 或兼容版本
- PyTorch
- NumPy
- h5py
- Matplotlib

可使用以下命令安装主要依赖：

```bash
pip install torch numpy h5py matplotlib
```

请根据本机 CUDA 版本，从 PyTorch 官方渠道安装对应的 PyTorch 构建。

## 准备数据集

使用 ModelNet40 HDF5 版本，并将解压后的文件放在默认目录：

```text
data/modelnet40_ply_hdf5_2048/modelnet40_ply_hdf5_2048/
```

目录中应至少包含：

```text
ply_data_train0.h5
ply_data_train1.h5
ply_data_train2.h5
ply_data_train3.h5
ply_data_train4.h5
ply_data_test0.h5
ply_data_test1.h5
```

数据集文件体积较大，不纳入 Git 版本管理。

## 训练

使用默认 baseline 配置：

```bash
python train_pointnet.py
```

显式指定参数：

```bash
python train_pointnet.py \
  --data-root "data/modelnet40_ply_hdf5_2048/modelnet40_ply_hdf5_2048" \
  --epoches 30 \
  --batch-size 16 \
  --num-points 1024 \
  --lr 0.001 \
  --seed 2026
```

PowerShell 中可使用反引号换行：

```powershell
python train_pointnet.py `
  --data-root "data/modelnet40_ply_hdf5_2048/modelnet40_ply_hdf5_2048" `
  --epoches 30 `
  --batch-size 16 `
  --num-points 1024 `
  --lr 0.001 `
  --seed 2026
```

程序会自动检测 CUDA；CUDA 可用时使用 GPU，否则使用 CPU。

## Quick Run

在正式训练前，可以使用少量训练和验证样本检查流程是否能正常运行：

```bash
python train_pointnet.py --quick-run
```

Quick Run 会使用 256 个输入点并训练 1 轮，仅用于检查代码，不应将其指标作为模型性能结果。

## 训练输出

默认输出保存在 `runs/`：

```text
runs/
├── best_pointnet.pt          # 验证集准确率最高的模型
├── last_checkpoint.pt        # 最后一轮训练状态
├── loss.png                  # 训练与验证 Loss 曲线
└── acc.png                   # 训练与验证准确率曲线
```

这些生成文件默认被 Git 忽略。如果需要长期保留实验结果，建议单独记录每次实验的 Git commit、参数和指标，而不是将大型权重直接提交到仓库。

## 后续计划

- 固定并保存分层训练/验证集索引
- 加入点云数据增强
- 增加学习率调度策略
- 完善命令行设备选择和实验配置记录
- 在统一评估流程下比较网络结构改进

