# Hyperspectral Data Loader

从 `.mat` 文件加载高光谱 / 多光谱数据，封装为 PyTorch Dataset，并提供波段可视化功能。

## 功能

- 兼容 MATLAB v5/v7 及 v7.3（HDF5）格式的 `.mat` 文件
- 自动检测变量名，无需手动指定
- 支持像素级（光谱向量）和 Patch 模式（适合 CNN 输入）
- 内置 min-max 归一化
- 波段图像可视化 & 单像素光谱曲线可视化
- 命令行直接运行

## 安装依赖

```bash
pip install torch scipy matplotlib numpy h5py
```

## 快速开始

### 作为模块导入

```python
from data_loader import MatDataset, visualize_band

# 加载数据集
dataset = MatDataset(
    "indian_pines.mat",
    data_key="data",    # 可省略，自动检测
    label_key="gt",     # 可省略，无标签时 label=-1
)

print(dataset)
# MatDataset(shape=(145, 145, 200), samples=21025, labels_unique=[0, 1, ..., 16])

# 取第 0 个像素
pixel, label = dataset[0]   # pixel: Tensor(200,), label: int

# 可视化第 30 个波段
visualize_band(dataset, band_index=30)
```

### Patch 模式（CNN 输入）

```python
dataset = MatDataset("data.mat", patch_size=9)
patch, label = dataset[0]   # patch: Tensor(Bands, 9, 9)
```

### DataLoader

```python
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
for x, y in loader:
    # x: (64, Bands) 或 (64, Bands, patch, patch)
    # y: (64,)
    ...
```

### 命令行

```bash
python data_loader.py --file indian_pines.mat --data-key data --band 30
# 带标签 + 保存图片
python data_loader.py --file data.mat --label-key gt --band 0 --save band0.png
```

**参数说明**

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--file` | `.mat` 文件路径（必填） | — |
| `--data-key` | 数据变量名 | 自动检测 |
| `--label-key` | 标签变量名 | 无（label=-1） |
| `--band` | 可视化的波段索引 | `0` |
| `--save` | 图片保存路径 | 弹窗显示 |

## API 参考

### `load_mat(filepath, key=None) → np.ndarray`

读取 `.mat` 文件中的数组。`key=None` 时自动选取第一个非元数据变量。

### `MatDataset`

| 参数 | 类型 | 说明 |
|---|---|---|
| `filepath` | str \| Path | `.mat` 文件路径 |
| `data_key` | str \| None | 数据变量名 |
| `label_key` | str \| None | 标签变量名 |
| `normalization` | bool | 是否做 min-max 归一化，默认 `True` |
| `patch_size` | int \| None | Patch 尺寸，`None` 返回像素向量 |

常用属性：`dataset.data`、`dataset.labels`、`dataset.height`、`dataset.width`、`dataset.num_bands`

### `visualize_band(dataset, band_index, cmap, title, save_path)`

可视化指定波段的二维灰度图。

### `visualize_spectrum(dataset, row, col, title)`

可视化指定像素的完整光谱曲线。

## 目录结构

```
data_loader/
├── data_loader.py   # 主模块
└── README.md
```

## License

MIT
