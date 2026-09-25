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

    # 数据和标签在同一个文件里
    dataset = MatDataset("data.mat", data_key="data", label_key="gt")

    # 数据和标签分处两个文件（Indian Pines / PaviaU / Salinas 都是这种布局）
    dataset = MatDataset(
        "Indian_pines.mat", data_key="indian_pines",
        label_file="Indian_pines_gt.mat", label_key="indian_pines_gt",
    )
    print(f"数据形状: {dataset.data.shape}")          # (H, W, Bands)
    print(f"样本数量: {len(dataset)}")                 # H * W

    pixel, label = dataset[0]                          # 取第 0 个像素
    visualize_band(dataset, band_index=30)             # 可视化第 30 个波段

    # 2. 直接运行脚本
    #    python data_loader.py --file data.mat --data-key data --band 30
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# HDF5 文件头魔数（MATLAB v7.3 的 .mat 就是 HDF5）
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


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
        第一个非元数据变量（跳过 '__' 和 '#' 开头的键）。

    Returns
    -------
    np.ndarray
        float32 数组。若文件本身就是 float32，则不会额外复制。
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    # 直接按文件头判断格式：v7.3 是 HDF5，其余交给 scipy。
    # 比依赖 scipy 抛出的异常类型更可靠。
    if _is_hdf5(filepath):
        return _load_hdf5(filepath, key)

    try:
        import scipy.io as sio
        mat = sio.loadmat(str(filepath))
    except NotImplementedError:
        # 兜底：scipy 仍要求使用 HDF5 reader
        return _load_hdf5(filepath, key)

    # 键的处理必须放在 try 之外，否则这里的 KeyError 有被兜底分支吞掉的风险
    if key is None:
        key = _pick_key(mat.keys())
    if key not in mat:
        raise KeyError(f"变量 '{key}' 不在文件中。可用变量: {_user_keys(mat.keys())}")
    # asarray 在输入已是 float32 时不会复制；array 则总会复制一整个数据立方体
    return np.asarray(mat[key], dtype=np.float32)


def _load_hdf5(filepath: Path, key: Optional[str]) -> np.ndarray:
    """读取 MATLAB v7.3（HDF5）格式的 .mat 文件。"""
    import h5py

    with h5py.File(str(filepath), "r") as f:
        if key is None:
            key = _pick_key(f.keys())
        if key not in f:
            raise KeyError(f"变量 '{key}' 不在文件中。可用变量: {_user_keys(f.keys())}")
        # h5py 读到的数组维度顺序和 MATLAB 相反，整轴反转可还原 (H, W[, B])。
        # 反转结果是 F-contiguous 视图，转成 C-contiguous 以便后续按行切片 /
        # 原地归一化时保持连续。
        return np.ascontiguousarray(np.asarray(f[key], dtype=np.float32).T)


def _is_hdf5(filepath: Path) -> bool:
    """通过文件头魔数判断是否为 HDF5（MATLAB v7.3）格式。"""
    with open(filepath, "rb") as fh:
        return fh.read(len(_HDF5_MAGIC)) == _HDF5_MAGIC


def _pick_key(keys):
    user_keys = _user_keys(keys)
    if not user_keys:
        raise ValueError("文件中没有可用变量")
    return user_keys[0]


def _user_keys(keys):
    # 过滤 MATLAB 元数据（'__' 前缀）与 HDF5 内部组（'#refs#'、'#subsystem#'），
    # 否则 v7.3 文件的自动选键可能选中这些非数据对象
    return [k for k in keys if not k.startswith("__") and not k.startswith("#")]


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
    label_file : str | Path, optional
        标签所在的 .mat 文件。若为 None,则从 filepath 中读取标签。
        Indian Pines / PaviaU / Salinas 等数据集的标签存放在另一个文件，
        此时需要指定本参数。
    normalization : bool
        是否对数据做 min-max 归一化，默认 True。
    norm_scope : str
        归一化统计量的取值范围:
        - "labeled"（默认）: 只用标签 > 0 的像素，排除背景 / 未标注像素
        - "global": 整个数据立方体，与旧版本行为逐位一致
        - "per_band": 逐波段统计，各波段量级差异大时更合适
    norm_stats : dict, optional
        复用外部统计量,形如 {"min": ..., "max": ...}。测试集传入训练集的
        统计量可保证严格无泄漏。拟合结果始终保存在 self.norm_stats_。
    patch_size : int, optional
        若 > 1,则返回以每个像素为中心的 (patch_size, patch_size, Bands)
        邻域 patch(自动 zero-padding)。默认 None,返回像素向量。
        建议取奇数，偶数时像素不再位于 patch 正中心。
    """

    def __init__(
        self,
        filepath: str | Path,
        data_key: Optional[str] = None,
        label_key: Optional[str] = None,
        label_file: Optional[str | Path] = None,
        normalization: bool = True,
        norm_scope: str = "labeled",
        norm_stats: Optional[dict] = None,
        patch_size: Optional[int] = None,
    ):
        # ---------- 加载数据 ----------
        data: np.ndarray = load_mat(filepath, data_key)  # (H, W, B) 或 (H, W)

        if data.ndim not in (2, 3):
            raise ValueError(
                f"数据维度应为 2 或 3，实际为 {data.ndim}: {data.shape}"
            )
        # 单波段图像（灰度图：H×W）和多波段高光谱数据统一成 (H, W, B) 三维格式
        if data.ndim == 2:
            data = data[..., np.newaxis]

        self.height, self.width, self.num_bands = data.shape

        # ---------- 加载标签 ----------
        if label_key is not None:
            label_source = filepath if label_file is None else label_file
            self.labels: np.ndarray = load_mat(label_source, label_key).astype(np.int64)
            if self.labels.ndim == 3 and self.labels.shape[2] == 1:
                self.labels = self.labels[:, :, 0]
            if self.labels.shape != (self.height, self.width):
                raise ValueError(
                    f"标签形状 {self.labels.shape} 与数据 (H, W)="
                    f"({self.height}, {self.width}) 不匹配"
                )
        else:
            self.labels = np.full((self.height, self.width), -1, dtype=np.int64)

        # ---------- 归一化 ----------
        # 必须在 np.pad 之前做：否则填充的 0 会参与统计量，
        # 且背景归一化后就不再是 0。
        self._data = data
        if normalization:
            if norm_stats is None:
                self.norm_stats_ = self._compute_norm_stats(norm_scope)
            else:
                self.norm_stats_ = self._as_stats(norm_stats)
            self._apply_norm(self.norm_stats_)
        else:
            self.norm_stats_ = None
        self.norm_scope = norm_scope

        # ---------- patch 模式 ----------
        self.patch_size = patch_size
        if patch_size is not None and patch_size > 1:
            if patch_size % 2 == 0:
                warnings.warn(
                    f"patch_size={patch_size} 为偶数，中心像素不在 patch 正中，"
                    "标签与 patch 中心会偏移半个像素；建议取奇数。",
                    stacklevel=2,
                )
            margin = patch_size // 2
            self.data_padded = np.pad(
                self._data,
                ((margin, margin), (margin, margin), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            # data 暴露为 padded 的视图，只保留一份缓冲区。
            # 视图带 margin 偏移，按未填充坐标索引即可。
            self._data = self.data_padded[
                margin : margin + self.height,
                margin : margin + self.width,
            ]
        else:
            self.data_padded = None

    @property
    def data(self) -> np.ndarray:
        """
        (H, W, B) 数据视图（patch 模式下是 data_padded 的切片，只读索引用途）。

        注意：这是读写视图，修改它会同时影响与之重叠的所有 patch。
        需要独立副本时请用 np.array(dataset.data)。
        """
        return self._data

    # ---------- 归一化辅助 ----------
    def _as_stats(self, stats: dict) -> dict:
        """把 {min, max} 统一广播成形状 (B,) 的 float32 数组。"""
        out = {}
        for name in ("min", "max"):
            if name not in stats:
                raise ValueError(f"norm_stats 缺少键 '{name}'")
            arr = np.asarray(stats[name], dtype=np.float32)
            out[name] = np.broadcast_to(arr, (self.num_bands,)).copy()
        return out

    def _compute_norm_stats(self, scope: str) -> dict:
        """按 scope 计算 min/max，统一返回形状 (B,) 的数组。"""
        if scope == "global":
            mn = self._data.min()
            mx = self._data.max()
            return self._as_stats({"min": mn, "max": mx})
        if scope == "per_band":
            return {
                "min": self._data.min(axis=(0, 1)).astype(np.float32),
                "max": self._data.max(axis=(0, 1)).astype(np.float32),
            }
        if scope == "labeled":
            mask = self.labels > 0
            if mask.any():
                labeled = self._data[mask]  # (N, B)
                return {
                    "min": labeled.min(axis=0).astype(np.float32),
                    "max": labeled.max(axis=0).astype(np.float32),
                }
            # 没有标签像素（例如 label_key=None），回落到 global
            return self._compute_norm_stats("global")
        raise ValueError(
            f"未知的 norm_scope: {scope!r}，可选 'labeled' / 'global' / 'per_band'"
        )

    def _apply_norm(self, stats: dict) -> None:
        """原地应用 min-max 归一化，避免额外分配两个完整数据立方体。"""
        mn, mx = stats["min"], stats["max"]
        rng = mx - mn
        # 完整的原地运算：self._data 来自 load_mat，是私有且未被别名引用的缓冲区
        np.subtract(self._data, mn, out=self._data)
        np.divide(self._data, np.where(rng > 0, rng, 1.0), out=self._data)

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
            ]  # (P, P, B) 视图
            # 一次拷贝同时完成转置和连续化：(B, P, P) 且 contiguous，
            # 下游 x.view(...) 不会报错；拷贝也切断了样本对大缓冲的引用。
            tensor = torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1)))
        else:
            # 拷贝很小（B*4 字节），保持返回的张量不别名数据集
            tensor = torch.from_numpy(self._data[row, col, :].copy())

        return tensor, label

    def get_band(self, band_index: int) -> np.ndarray:
        """返回第 band_index 个波段的 2D 图像 (H, W)。"""
        if band_index < 0 or band_index >= self.num_bands:
            raise IndexError(
                f"波段索引 {band_index} 超出范围 [0, {self.num_bands - 1}]"
            )
        return self._data[:, :, band_index]

    def __repr__(self) -> str:
        scope = self.norm_scope if self.norm_stats_ is not None else None
        return (
            f"MatDataset(shape=({self.height}, {self.width}, {self.num_bands}), "
            f"samples={len(self)}, patch_size={self.patch_size}, "
            f"norm_scope={scope!r}, labels_unique={np.unique(self.labels).tolist()})"
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
    save_path: Optional[str] = None,
) -> None:
    """
    可视化某个像素的完整光谱曲线。

    Parameters
    ----------
    dataset : MatDataset
    row, col : int
        像素坐标。
    title : str, optional
        图片标题。
    save_path : str, optional
        如指定，将图片保存到该路径；否则弹窗显示。
    """
    spectrum = dataset.data[row, col, :]
    label = dataset.labels[row, col]
    normalized = dataset.norm_stats_ is not None

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(range(len(spectrum)), spectrum, linewidth=1)
    ax.set_xlabel("波段序号")
    ax.set_ylabel("反射率 (归一化)" if normalized else "反射率 (原始值)")
    ax.set_title(title or f"像素 ({row}, {col})  类别={label}")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图片已保存: {save_path}")
    else:
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
    parser.add_argument(
        "--label-file", default=None, help="标签所在的 .mat 文件（与数据分开时使用）"
    )
    parser.add_argument("--band", type=int, default=0, help="要可视化的波段索引")
    parser.add_argument("--row", type=int, default=None, help="光谱曲线像素行号")
    parser.add_argument("--col", type=int, default=None, help="光谱曲线像素列号")
    parser.add_argument(
        "--norm-scope",
        default="labeled",
        choices=["labeled", "global", "per_band"],
        help="归一化统计范围",
    )
    parser.add_argument("--save", default=None, help="保存可视化图片路径")
    args = parser.parse_args()

    dataset = MatDataset(
        filepath=args.file,
        data_key=args.data_key,
        label_key=args.label_key,
        label_file=args.label_file,
        norm_scope=args.norm_scope,
    )
    print(dataset)

    # 演示 DataLoader
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    batch_x, batch_y = next(iter(loader))
    print(f"Batch 形状: x={batch_x.shape}, y={batch_y.shape}")

    # 可视化指定波段
    visualize_band(dataset, band_index=args.band, save_path=args.save)

    # 指定了行列则额外画一条光谱曲线。
    # 已有 --save 时加后缀，避免覆盖上面保存的波段图。
    if args.row is not None and args.col is not None:
        spectrum_path = args.save
        if args.save is not None:
            p = Path(args.save)
            spectrum_path = str(p.with_name(f"{p.stem}_spectrum{p.suffix or '.png'}"))
        visualize_spectrum(
            dataset, row=args.row, col=args.col, save_path=spectrum_path
        )


if __name__ == "__main__":
    main()
