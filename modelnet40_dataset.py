import h5py
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


# 构建数据集
class ModelNet40(Dataset):
    def __init__(self, root, split='train', num_points=1024):
        super().__init__()
        root = Path(root)

        assert split in ("train","test") # 这是什么语法？
        self.num_points = num_points

        # 搜索该划分下所有HDF5文件
        files= root.glob(f'ply_data_{split}*.h5')
        if not files:
            raise FileNotFoundError(f"未找到{split}HDF5文件：{root}")

        data_list = []
        label_list = []
        # 读取所有数据分片
        for path in files:
            with h5py.File(path, 'r') as f:
                data_list.append(f['data'][:].astype(np.float32))
                label_list.append(f['label'][:].reshape(-1)) # 为什么要reshape
        # [M, 2048, 3]
        self.data = np.concatenate(data_list,axis=0)
        self.labels = np.concatenate(label_list,axis=0).astype(np.int16)

        assert 1<=num_points<self.data.shape[1] # 这是什么意思？



    def __getitem__(self, item):
        # 取出一份点云数据
        # self.data :[M, 2048, 3] ->[num_pints, 3] （points)
        points = self.data[item,:self.num_points].copy()
        label = self.labels[item]
        points = torch.tensor(points) # 转成张量形式
        label = torch.tensor(label, dtype=torch.long) # 转成张量形式
        return points, label



    def __len__(self):
        return len(self.labels)