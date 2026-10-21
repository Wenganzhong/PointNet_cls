from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ModelNet40(Dataset):
    def __init__(
        self,
        root,
        split="train",
        num_points=1024
    ):
        super().__init__()

        root = Path(root)
        assert split in ("train", "test")

        self.num_points = num_points

        # 搜索该划分下所有 HDF5 数据文件
        files = sorted(
            root.glob(f"ply_data_{split}*.h5")
        )

        if not files:
            raise FileNotFoundError(
                f"未找到 {split} HDF5 文件：{root}"
            )

        data_list = []
        label_list = []

        # 读取所有数据分片
        for path in files:
            with h5py.File(path, "r") as f:
                data_list.append(
                    f["data"][:].astype(np.float32)
                )
                label_list.append(
                    f["label"][:].reshape(-1)
                )

        # [M, 2048, 3]
        self.data = np.concatenate(
            data_list, axis=0
        )

        # [M]
        self.labels = np.concatenate(
            label_list, axis=0
        ).astype(np.int64)

        assert 1 <= num_points <= self.data.shape[1]

        print(
            f"{split} 数据集：{len(self.data)} 个样本"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # 取出一份物体点云
        # [2048, 3] -> [N, 3]
        points = self.data[idx, :self.num_points].copy()

        # 当前物体的类别编号
        label = self.labels[idx]

        # 转换为 PyTorch Tensor
        points = torch.from_numpy(points)
        label = torch.tensor(
            label, dtype=torch.long
        )

        return points, label


if __name__ == "__main__":

    dataset = ModelNet40(
        root="data/modelnet40_ply_hdf5_2048",
        split="train",
        num_points=1024
    )

    points, label = dataset[0]

    print("单个样本点云：", points.shape)
    print("单个样本标签：", label.shape)
    print("标签值：", label.item())
