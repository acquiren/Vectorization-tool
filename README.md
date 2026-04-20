# 图像矢量化工具

[![Python](https://img.shields.io/badge/Python-3.8.20-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1%2Bcpu-orange.svg)](https://pytorch.org/)
[![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)](https://riverbankcomputing.com/software/pyqt/)
[![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)]()

将位图（PNG/JPG）转换为可缩放矢量图（SVG）的桌面应用，集成了多种矢量化算法，并提供友好的图形界面。本项目基于 [SuperSVG (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Hu_SuperSVG_Superpixel-based_Scalable_Vector_Graphics_Synthesis_CVPR_2024_paper.pdf) 论文实现，为安徽大学本科毕业设计作品。

---

## 目录

- [功能特性](#功能特性)
- [系统架构](#系统架构)
- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
- [使用说明](#使用说明)
  - [启动图形界面](#启动图形界面)
  - [基础模式](#基础模式)
  - [高级模式](#高级模式)
  - [Python API 调用](#python-api-调用)
- [模块说明](#模块说明)
- [项目结构](#项目结构)
- [参考文献](#参考文献)

---

## 功能特性

- 🎨 **Potrace 黑白矢量化** — 基于传统轮廓追踪算法，将灰度图像转换为精确的 SVG 路径
- 🌈 **Color Trace 彩色矢量化** — 多颜色分层描摹，支持最多 256 色量化与自定义调色板
- 🤖 **SuperSVG 深度学习矢量化** — 基于超像素分割的神经网络推理，输出高质量笔画风格 SVG
- ✨ **DiffVG 后处理优化** — 可微分渲染优化器，通过梯度下降进一步提升矢量图视觉还原度
- 🖥️ **双模式 GUI** — 基础模式（一键操作）与高级模式（精细参数调节），支持拖拽导入图片
- 📊 **实时预览** — 输入与输出图像并排显示，矢量化结果即时渲染

---

## 系统架构

```
输入图片 (PNG/JPG)
      │
      ├──► Potrace 黑白矢量化 ──────────────┐
      │         └─ 二值化 → Potrace → SVG   │
      │                                      │
      ├──► Color Trace 彩色矢量化 ──────────►├──► DiffVG 后处理优化 ──► 最终 SVG
      │         └─ 颜色量化 → 分层描摹 → SVG │         └─ Adam优化器 (MSE/LPIPS)
      │                                      │
      └──► SuperSVG 深度学习矢量化 ──────────┘
                └─ 神经网络推理 → 笔画参数 → SVG
```

---

## 环境要求

| 依赖项 | 版本 |
|--------|------|
| Python | 3.8.20 |
| PyTorch | 1.13.1（CPU Only） |
| PyQt5 | ≥ 5.15 |
| pydiffvg | 需从源码编译 |
| scikit-image | ≥ 0.19 |
| Pillow | ≥ 9.0 |
| numpy | ≥ 1.23 |
| torchvision | 对应 PyTorch 1.13.1 |

**内置工具（无需另行安装）：**

- `tools/potrace/potrace.exe` — Potrace 矢量描摹引擎
- `tools/pngquant/pngquant.exe` — PNG 颜色量化工具
- `tools/ImageMagick/magick.exe` — 图像格式转换工具

---

## 安装步骤

### 1. 克隆仓库

```bash
git clone <repository-url>
cd project
```

### 2. 创建 Conda 环境

```bash
conda create -n vectorizer python=3.8.20
conda activate vectorizer
```

### 3. 安装 Python 依赖

```bash
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install PyQt5 scikit-image Pillow numpy
```

### 4. 安装 pydiffvg

```bash
cd DiffVG
git submodule update --init --recursive
python setup.py install
cd ..
```

### 5. 下载预训练模型

将 SuperSVG 预训练检查点下载并放置到：

```
SuperSVG/coarse-model.pt
```

> 下载链接：[Google Drive](https://drive.google.com/file/d/10C9EsMD6_B7dCEz6oNJetevNazk3blBK/view?usp=drive_link)

---

## 使用说明

### 启动图形界面

```bash
python vectorizer_app.py
```

### 基础模式

启动后默认进入基础模式，操作流程：

1. 点击「导入图片」或直接拖拽图片到窗口（支持 PNG/JPG/BMP）
2. 选择矢量化方式：
   - **启用色彩保留** — 使用彩色 Potrace，可调整颜色数（1-256）
   - **启用细节优化** — 使用 DiffVG 后处理，可调整迭代次数
   - **启用 SuperSVG** — 使用深度学习矢量化（与上两项互斥）
3. 点击「开始矢量化」
4. 输出 SVG 预览将自动显示在右侧

### 高级模式

切换到高级模式可对每个算法进行精细参数调节：

**Potrace 参数：**

| 参数 | 含义 | 默认值 |
|------|------|--------|
| 噪点抑制 (turdsize) | 忽略面积小于此值的斑点 | 2 |
| 平滑度 (alphamax) | 转角平滑强度 (0~1.334) | 1.0 |
| 曲线优化容差 (opttolerance) | 贝塞尔曲线优化精度 | 0.3 |
| 转角策略 (turnpolicy) | 歧义像素处理策略 | minority |

**彩色矢量化参数：**

| 参数 | 含义 | 默认值 |
|------|------|--------|
| 颜色数 | 量化目标颜色数 | 8 |
| 量化算法 | mc（中切）/ as（自适应） | mc |
| 去斑点 | 噪点抑制像素数 | 2 |

**SuperSVG 参数：**

| 参数 | 含义 | 默认值 |
|------|------|--------|
| 笔画数 (stroke_num) | SVG 路径总数 | 128 |
| 路径段数 (path_num) | 每条笔画的段数 | 4 |
| 微调迭代次数 | 后处理优化轮次（0=跳过） | 50 |
| 路径学习率 | Adam 优化器路径点步长 | 1.0 |
| 颜色学习率 | Adam 优化器颜色步长 | 0.01 |

### Python API 调用

#### SuperSVG 推理

```python
from SuperSVG.inference import bitmap_to_svg

svg_object, svg_path = bitmap_to_svg(
    input_image_path=r"test_image/6.jpg",
    checkpoint_path=r"SuperSVG/coarse-model.pt",
    output_svg_path=r"output/out.svg",  # 可选，不传则与输入同名
    device="cpu",
    stroke_num=128,          # 笔画数量
    path_num=4,              # 每笔画段数
    finetune_iter=200,       # 微调迭代次数，0 则不微调
    lr_path=1.0,             # 路径学习率
    lr_color=0.01,           # 颜色学习率
)
print(f"SVG 已保存至: {svg_path}")
```

#### Potrace 彩色矢量化

```python
from Potrace.color_trace import color_trace

color_trace(
    input_list=["test_image/1.png"],
    output_list=["output/out.svg"],
    color_count=8,           # 颜色数量
    process_count=1,         # 处理进程数
    quantization="mc",       # 量化算法: mc / as / nq
    dither=None,             # 拟色算法: None / floydsteinberg / riemersma
    despeckle=2,             # 去斑点阈值
    smoothcorners=1.0,       # 转角平滑
    optimizepaths=0.2,       # 路径优化容差
)
```

#### DiffVG 后处理优化

```python
from DiffVG.refine_svg import refine_svg

refine_svg(
    svg_path="output/out.svg",
    png_path="test_image/1.png",
    use_lpips_loss=False,    # True 使用感知损失，False 使用 MSE
    num_iter=50,             # 优化迭代次数
)
# 优化结果保存在 ./tmp/refine_svg/iter_{num_iter-1}.svg
```

---

## 模块说明

| 模块 | 路径 | 说明 |
|------|------|------|
| 主应用 | `vectorizer_app.py` | PyQt5 GUI 入口，App 类 |
| SuperSVG 推理 | `SuperSVG/inference.py` | `bitmap_to_svg()` 函数 |
| SuperSVG 模型 | `SuperSVG/models/supersvg_coarse.py` | 粗粒度阶段网络结构 |
| SVG 渲染工具 | `SuperSVG/util/SVR_render.py` | SVGObject 类，笔画渲染 |
| 彩色描摹 | `Potrace/color_trace.py` | `color_trace()` 函数 |
| 黑白描摹 | `Potrace/potrace_vector.py` | `PotraceRunner` 类 |
| DiffVG 优化 | `DiffVG/refine_svg.py` | `refine_svg()` 函数 |
| SVG 合并 | `Potrace/svg_stack/` | 多色层 SVG 合并工具 |
| UI 界面 | `ui/vectorizer_ui.ui` | Qt Designer 界面文件 |

---

## 项目结构

```
project/
├── vectorizer_app.py          # 主应用入口 (PyQt5)
├── CODEBUDDY.md               # 项目开发指南
├── 开发日志.md                 # 开发进度记录
│
├── SuperSVG/                  # 深度学习矢量化模块
│   ├── inference.py           # 推理接口 (bitmap_to_svg)
│   ├── main_coarse.py         # 粗阶段训练入口
│   ├── engine_coarse.py       # 训练引擎
│   ├── coarse-model.pt        # 预训练模型权重
│   ├── models/                # 神经网络模型
│   │   ├── supersvg_coarse.py # 粗阶段模型
│   │   ├── encoder.py         # 图像编码器
│   │   └── morphology.py      # 形态学处理
│   └── util/                  # 工具库
│       ├── SVR_render.py      # SVG 渲染
│       ├── cross_attention.py # 交叉注意力
│       └── dpw.py             # 动态路径权重
│
├── Potrace/                   # 传统矢量化模块
│   ├── color_trace.py         # 彩色矢量描摹 (color_trace)
│   ├── potrace_vector.py      # 黑白矢量化 (PotraceRunner)
│   └── svg_stack/             # SVG 图层合并工具
│
├── DiffVG/                    # 可微分渲染优化模块
│   └── refine_svg.py          # SVG 后处理优化 (refine_svg)
│
├── tools/                     # 内置可执行工具
│   ├── potrace/               # Potrace 描摹引擎
│   ├── pngquant/              # PNG 颜色量化
│   └── ImageMagick/           # 图像处理套件
│
├── ui/
│   └── vectorizer_ui.ui       # Qt Designer 界面定义
│
├── test_image/                # 测试图片
├── output/                    # SVG 输出目录
└── tmp/                       # 临时文件目录
```

---

## 参考文献

- **SuperSVG**: Hu T, Yi R, et al. "SuperSVG: Superpixel-based Scalable Vector Graphics Synthesis." *CVPR 2024*. [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Hu_SuperSVG_Superpixel-based_Scalable_Vector_Graphics_Synthesis_CVPR_2024_paper.pdf)
- **DiffVG**: Li T M, et al. "Differentiable Vector Graphics Rasterization for Editing and Learning." *ACM SIGGRAPH Asia 2020*. [[GitHub]](https://github.com/BachiLi/diffvg)
- **Potrace**: Selinger P. "Potrace: a polygon-based tracing algorithm." *2003*. [[Homepage]](http://potrace.sourceforge.net/)
