# Hyperspectral Data Loader

从 `.mat` 文件加载高光谱 / 多光谱数据，封装为 PyTorch Dataset，并提供波段可视化功能。

## 功能

- 兼容 MATLAB v5/v7 及 v7.3（HDF5）格式的 `.mat` 文件（按文件头魔数自动判别）
- 自动检测变量名，无需手动指定
- 支持数据和标签分处两个文件的数据集（Indian Pines / PaviaU / Salinas）
- 支持像素级（光谱向量）和 Patch 模式（适合 CNN 输入）
- 内置 min-max 归一化，统计范围可配置（默认排除背景像素，避免统计量泄漏）
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

# 数据和标签在同一个文件里
dataset = MatDataset(
    "data.mat",
    data_key="data",    # 可省略，自动检测
    label_key="gt",     # 可省略，无标签时 label=-1
)

# 数据和标签分处两个文件（Indian Pines / PaviaU / Salinas 都是这种布局）
dataset = MatDataset(
    "Indian_pines.mat",
    data_key="indian_pines",
    label_file="Indian_pines_gt.mat",
    label_key="indian_pines_gt",
)

print(dataset)
# MatDataset(shape=(145, 145, 220), samples=21025, patch_size=None,
#            norm_scope='labeled', labels_unique=[0, 1, ..., 16])

# 取第 0 个像素
pixel, label = dataset[0]   # pixel: Tensor(220,), label: int

# 可视化第 30 个波段
visualize_band(dataset, band_index=30)
```

### 归一化

`normalization=True`（默认）时做 min-max 归一化，统计范围由 `norm_scope` 决定：

| `norm_scope` | 说明 |
|---|---|
| `"labeled"`（默认） | 只用标签 > 0 的像素，排除背景 / 未标注像素（避免统计量泄漏） |
| `"global"` | 整个数据立方体，与旧版本行为逐位一致 |
| `"per_band"` | 逐波段统计，各波段量级差异大时更合适 |

要让测试集严格无泄漏，可复用训练集的统计量：

```python
train_ds = MatDataset("train.mat", data_key="X", label_key="gt")
test_ds  = MatDataset("test.mat",  data_key="X", label_key="gt",
                      norm_stats=train_ds.norm_stats_)   # 应用训练集统计量
```

### Patch 模式（CNN 输入）

```python
dataset = MatDataset("data.mat", patch_size=9)
patch, label = dataset[0]   # patch: Tensor(Bands, 9, 9)，连续内存
```

> `patch_size` 建议取**奇数**，否则中心像素不在 patch 正中，标签会偏移半个像素。
> patch 模式下 `dataset.data` 是 `data_padded` 的视图（只保留一份缓冲区）；
> 它是读写视图，修改会影响与之重叠的所有 patch，需要独立副本请用 `np.array(dataset.data)`。

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
python data_loader.py --file data.mat --data-key data --band 30
# 标签在另一个文件 + 保存图片
python data_loader.py --file Indian_pines.mat --data-key indian_pines \
    --label-file Indian_pines_gt.mat --label-key indian_pines_gt \
    --band 30 --save band30.png
# 额外画一条 (50, 50) 处的光谱曲线（会存成 band30_spectrum.png）
python data_loader.py --file data.mat --data-key data --band 30 --row 50 --col 50 \
    --save band30.png
```

**参数说明**

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--file` | `.mat` 文件路径（必填） | — |
| `--data-key` | 数据变量名 | 自动检测 |
| `--label-key` | 标签变量名 | 无（label=-1） |
| `--label-file` | 标签所在的 `.mat` 文件 | 与数据同文件 |
| `--band` | 可视化的波段索引 | `0` |
| `--row` `--col` | 额外绘制该像素的光谱曲线 | 不绘制 |
| `--norm-scope` | 归一化统计范围 | `labeled` |
| `--save` | 图片保存路径 | 弹窗显示 |

> 同时给出 `--row/--col` 和 `--save` 时会写出两张图（波段图 + 光谱图，后者加 `_spectrum` 后缀）。
> 在有中文字体缺失的机器上保存图片可能提示 Glyph missing，属 Matplotlib 字体问题，不影响数据。

## API 参考

### `load_mat(filepath, key=None) → np.ndarray`

读取 `.mat` 文件中的数组。`key=None` 时自动选取第一个非元数据变量。

### `MatDataset`

| 参数 | 类型 | 说明 |
|---|---|---|
| `filepath` | str \| Path | `.mat` 文件路径 |
| `data_key` | str \| None | 数据变量名 |
| `label_key` | str \| None | 标签变量名 |
| `label_file` | str \| Path \| None | 标签所在文件，`None` 表示与数据同文件 |
| `normalization` | bool | 是否做 min-max 归一化，默认 `True` |
| `norm_scope` | str | `"labeled"`（默认）/ `"global"` / `"per_band"` |
| `norm_stats` | dict \| None | 复用外部 `{"min", "max"}` 统计量 |
| `patch_size` | int \| None | Patch 尺寸，`None` 返回像素向量 |

常用属性：`dataset.data`、`dataset.labels`、`dataset.height`、`dataset.width`、
`dataset.num_bands`、`dataset.norm_stats_`（拟合后的统计量，可传给测试集）。

标签形状与数据不一致、数据维度不是 2/3、或 `norm_scope` 取值非法时，
构造 `MatDataset` 会直接抛出带说明的 `ValueError`。

### `visualize_band(dataset, band_index, cmap, title, save_path)`

可视化指定波段的二维灰度图。

### `visualize_spectrum(dataset, row, col, title, save_path)`

可视化指定像素的完整光谱曲线。

## 目录结构

```
data_loader/
├── data_loader.py   # 主模块
└── README.md
```

## License

MIT
