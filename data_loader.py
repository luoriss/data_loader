"""
data_loader.py
==============
从 .mat 文件加载高光谱/多光谱数据，封装为 PyTorch Dataset，
并提供单波段可视化功能。

依赖:
    pip install torch scipy matplotlib numpy h5py

用法:
    # 1. 作为模块导入
    from data_loader import MatDataset, visualize_band

    dataset = MatDataset("indian_pines.mat", data_key="data", label_key="gt")
    print(f"数据形状: {dataset.data.shape}")          # (H, W, Bands)
    print(f"样本数量: {len(dataset)}")                 # H * W

    pixel, label = dataset[0]                          # 取第 0 个像素
    visualize_band(dataset, band_index=30)             # 可视化第 30 个波段

    # 2. 直接运行脚本
    #    python data_loader.py --file data.mat --data-key data --band 30
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────
# 1. 读取 .mat 文件（兼容 v5 和 v7.3 格式）
# ──────────────────────────────────────────────
def load_mat(filepath: str | Path, key: Optional[str] = None) -> np.ndarray:
    """
    读取 .mat 文件中的数组。

    Parameters
    ----------
    filepath : str | Path
        .mat 文件路径。
    key : str, optional
        要读取的变量名。若为 None，则自动选择文件中
        第一个非元数据变量（跳过 '__' 开头的键）。

    Returns
    -------
    np.ndarray
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    # 先尝试 scipy（MATLAB v5 / v7）
    try:
        import scipy.io as sio
        #sio返回python字典，键是MATLAB变量名，值是对应的numpy数组
        #mat为键值字典，如果key为None，则自动选择第一个非元数据变量
        mat = sio.loadmat(str(filepath))
        
        if key is None:
            # 自动找第一个有意义的变量
            key = _pick_key(mat.keys())
        if key not in mat:
            raise KeyError(f"变量 '{key}' 不在文件中。可用变量: {_user_keys(mat.keys())}")
        return np.array(mat[key], dtype=np.float32)

    except NotImplementedError:
        # v7.3 格式，用 h5py 读取
        import h5py

        with h5py.File(str(filepath), "r") as f:
            if key is None:
                key = _pick_key(f.keys())
            if key not in f:
                raise KeyError(f"变量 '{key}' 不在文件中。可用变量: {list(f.keys())}")
            # h5py 读到的数组维度顺序和 MATLAB 相反，需要转置
            return np.array(f[key], dtype=np.float32).T


def _pick_key(keys):
    user_keys = _user_keys(keys)
    if not user_keys:
        raise ValueError("文件中没有可用变量")
    return user_keys[0]


def _user_keys(keys):
    return [k for k in keys if not k.startswith("__")]


# ──────────────────────────────────────────────
# 2. PyTorch Dataset
# ──────────────────────────────────────────────
class MatDataset(Dataset):
    """
    将 .mat 高光谱/多光谱数据封装为 PyTorch Dataset。

    每个样本是一个像素向量 (Bands,)，标签为对应的整数类别。
    如果没有提供标签，则标签统一为 -1。

    Parameters
    ----------
    filepath : str | Path
        .mat 文件路径。
    data_key : str, optional
        数据变量名（自动检测时可省略）。
    label_key : str, optional
        标签变量名（可省略，无标签时 label=-1)。
    normalization : bool
        是否对每个像素做 min-max 归一化，默认 True。
    patch_size : int, optional
        若 > 1,则返回以每个像素为中心的 (patch_size, patch_size, Bands)
        邻域 patch(自动 zero-padding)。默认 None,返回像素向量。
    """

    def __init__(
        self,
        filepath: str | Path,
        data_key: Optional[str] = None,
        label_key: Optional[str] = None,
        normalization: bool = True,
        patch_size: Optional[int] = None,
    ):
        # ---------- 加载数据 ----------
        self.data: np.ndarray = load_mat(filepath, data_key)  # (H, W, B) 或 (H, W)
        
        #单波段图像（灰度图：H×W）和多波段高光谱数据统一成 (H, W, B) 三维格式
        if self.data.ndim == 2:
            # 单波段数据，增加一个维度
            self.data = self.data[..., np.newaxis]
        
        self.height, self.width, self.num_bands = self.data.shape

        # ---------- 加载标签 ----------
        if label_key is not None:
            self.labels: np.ndarray = load_mat(filepath, label_key).astype(np.int64)
            if self.labels.ndim == 3:
                self.labels = self.labels[:, :, 0]
        else:
            self.labels = np.full((self.height, self.width), -1, dtype=np.int64)

        # ---------- 归一化(高斯正态分布) ----------
        if normalization:
            dmin = self.data.min()
            dmax = self.data.max()
            if dmax - dmin > 0:
                self.data = (self.data - dmin) / (dmax - dmin)

        # ---------- patch 模式 ----------
        self.patch_size = patch_size
        if patch_size is not None and patch_size > 1:
            margin = patch_size // 2
            self.data_padded = np.pad(
                self.data,
                ((margin, margin), (margin, margin), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        else:
            self.data_padded = None

    # ---------- Dataset 接口 ----------
    def __len__(self) -> int:
        return self.height * self.width

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = index // self.width
        col = index % self.width
        label = int(self.labels[row, col])

        if self.data_padded is not None:
            patch_size = self.patch_size
            if patch_size is None:
                raise ValueError("patch_size must not be None when data_padded is set")
            patch = self.data_padded[
                row : row + patch_size,
                col : col + patch_size,
                :,
            ]
            # (patch, patch, B) → (B, patch, patch)  适合 CNN
            tensor = torch.from_numpy(patch.copy()).permute(2, 0, 1)
        else:
            tensor = torch.from_numpy(self.data[row, col, :].copy())

        return tensor, label

    def get_band(self, band_index: int) -> np.ndarray:
        """返回第 band_index 个波段的 2D 图像 (H, W)。"""
        if band_index < 0 or band_index >= self.num_bands:
            raise IndexError(
                f"波段索引 {band_index} 超出范围 [0, {self.num_bands - 1}]"
            )
        return self.data[:, :, band_index]

    def __repr__(self) -> str:
        return (
            f"MatDataset(shape=({self.height}, {self.width}, {self.num_bands}), "
            f"samples={len(self)}, labels_unique={np.unique(self.labels).tolist()})"
        )


# ──────────────────────────────────────────────
# 3. 可视化
# ──────────────────────────────────────────────
def visualize_band(
    dataset: MatDataset,
    band_index: int = 0,
    cmap: str = "viridis",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    可视化数据集中任意一个光谱波段。

    Parameters
    ----------
    dataset : MatDataset
    band_index : int
        波段索引（从 0 开始）。
    cmap : str
        Matplotlib 色图名称。
    title : str, optional
        图片标题。
    save_path : str, optional
        如指定，将图片保存到该路径。
    """
    band_img = dataset.get_band(band_index)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(band_img, cmap=cmap)
    ax.set_title(title or f"Band #{band_index}  (共 {dataset.num_bands} 个波段)")
    ax.set_xlabel("列 (x)")
    ax.set_ylabel("行 (y)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图片已保存: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def visualize_spectrum(
    dataset: MatDataset,
    row: int,
    col: int,
    title: Optional[str] = None,
) -> None:
    """
    可视化某个像素的完整光谱曲线。

    Parameters
    ----------
    dataset : MatDataset
    row, col : int
        像素坐标。
    """
    spectrum = dataset.data[row, col, :]
    label = dataset.labels[row, col]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(range(len(spectrum)), spectrum, linewidth=1)
    ax.set_xlabel("波段序号")
    ax.set_ylabel("反射率 (归一化)")
    ax.set_title(title or f"像素 ({row}, {col})  类别={label}")
    plt.tight_layout()
    plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────
# 4. 命令行入口
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="加载 .mat 高光谱数据并可视化")
    parser.add_argument("--file", required=True, help=".mat 文件路径")
    parser.add_argument("--data-key", default=None, help="数据变量名（可自动检测）")
    parser.add_argument("--label-key", default=None, help="标签变量名")
    parser.add_argument("--band", type=int, default=0, help="要可视化的波段索引")
    parser.add_argument("--save", default=None, help="保存可视化图片路径")
    args = parser.parse_args()

    dataset = MatDataset(
        filepath=args.file,
        data_key=args.data_key,
        label_key=args.label_key,
    )
    print(dataset)

    # 演示 DataLoader
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    batch_x, batch_y = next(iter(loader))
    print(f"Batch 形状: x={batch_x.shape}, y={batch_y.shape}")

    # 可视化指定波段
    visualize_band(dataset, band_index=args.band, save_path=args.save)


if __name__ == "__main__":
    main()
