#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperSVG 推理脚本

用于将 PNG/JPG 位图转换为 SVG 矢量图，并支持多次迭代微调优化
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# 关键：将 `SuperSVG/` 目录加入 sys.path，兼容项目内大量 `from util...` / `from models...` 的导入写法
_SUPERSVG_DIR = Path(__file__).resolve().parent  # 当前文件所在目录（即 SuperSVG/）
if str(_SUPERSVG_DIR) not in sys.path:  # 避免重复插入
    sys.path.insert(0, str(_SUPERSVG_DIR))  # 让 `util`、`models` 可作为顶层模块被找到

# 导入项目中的模块
from models.supersvg_coarse import SuperSVG_coarse
from util import SVR_render
import pydiffvg


def parse_args():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description='SuperSVG 推理：将位图转换为矢量图')
    parser.add_argument('--input', type=str, required=True,
                        help='输入图片路径（PNG/JPG格式）')
    parser.add_argument('--output', type=str, default='output.svg',
                        help='输出SVG文件路径（默认为output.svg）')
    parser.add_argument('--checkpoint', type=str, default='coarse-model.pt',
                        help='预训练模型检查点路径（默认为coarse-model.pt）')
    parser.add_argument('--device', type=str, default='cpu',
                        help='运行设备（cpu/cuda），由于不支持cuda，默认使用cpu')
    parser.add_argument('--width', type=int, default=224,
                        help='输入和输出图像的宽度（默认为224）')
    parser.add_argument('--stroke_num', type=int, default=128,
                        help='笔画数量（默认为128）')
    parser.add_argument('--path_num', type=int, default=4,
                        help='每个笔画的路径数量（默认为4）')
    parser.add_argument('--finetune_iter', type=int, default=50,
                        help='微调迭代次数（默认为50，设置为0则不微调）')
    parser.add_argument('--lr_path', type=float, default=1.0,
                        help='路径点学习率（默认为1.0）')
    parser.add_argument('--lr_color', type=float, default=0.01,
                        help='颜色学习率（默认为0.01）')
    return parser.parse_args()


def load_image(image_path, target_size, device):
    """
    加载并预处理图片
    
    Args:
        image_path: 图片路径
        target_size: 目标尺寸（width, height）
        device: 运行设备
    
    Returns:
        tuple: (预处理后的图片张量, 原始图片numpy数组, 原始图片尺寸)
    """
    # 打开图片
    img_pil = Image.open(image_path).convert('RGB')
    original_size = img_pil.size  # 保存原始尺寸 (width, height)
    
    # 保存原始图片的numpy数组用于微调
    img_np = np.array(img_pil)
    # 如果原始尺寸不是target_size，调整大小
    if img_np.shape[0] != target_size or img_np.shape[1] != target_size:
        img_pil_resized = img_pil.resize((target_size, target_size))
        img_np = np.array(img_pil_resized)
        img_pil = img_pil_resized  # 同步用于张量预处理的 PIL 图像，避免张量仍取原尺寸
    
    # 预处理：与训练时保持一致
    transform = Compose([
        Resize((target_size, target_size)),  # 调整大小
        ToTensor(),  # 转换为张量
    ])
    
    img_tensor = transform(img_pil).unsqueeze(0).to(device)  # 添加批次维度并移动到设备
    return img_tensor, img_np, original_size


def load_model(checkpoint_path, device, width=224, stroke_num=128, path_num=4):
    """
    加载预训练模型
    
    Args:
        checkpoint_path: 检查点文件路径
        device: 运行设备
        width: 图像宽度
        stroke_num: 笔画数量
        path_num: 每个笔画的路径数量
    
    Returns:
        nn.Module: 加载好的模型
    """
    # 创建模型（使用与训练时相同的参数）
    model = SuperSVG_coarse(
        stroke_num=stroke_num,
        path_num=path_num,
        width=width,
        num_loss=True
    )
    
    # 加载检查点
    print(f'正在加载模型检查点: {checkpoint_path}')
    # 注意：这里我们直接加载state_dict，因为main_coarse.py中保存的是model.state_dict()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 检查是否是完整的检查点（包含model键）还是直接的state_dict
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()  # 设置为评估模式
    print('模型加载成功！')
    return model


def strokes_to_svg_object(strokes, width, device, path_num=4):
    """
    将笔画参数转换为SVGObject
    
    Args:
        strokes: 笔画参数
        width: SVG宽度
        device: 设备
        path_num: 每个笔画的路径数量
    
    Returns:
        SVGObject: SVG对象
    """
    # 创建SVGObject
    svg_object = SVR_render.SVGObject(size=(width, width))
    
    # 处理每个笔画
    strokes = strokes[0].cpu().numpy()  # 取第一个批次的数据并转为numpy
    num_strokes = strokes.shape[0]
    
    # 每个路径的控制点数量
    num_control_points = [2] * path_num
    
    shapes = []
    groups = []
    
    for num in range(num_strokes):
        # 获取当前笔画的参数
        stroke = strokes[num]
        
        # 提取路径点
        # 路径点格式：每个路径有3个点（起点，控制点1，控制点2），所以 path_num*3*2 个坐标
        path_points = stroke[:-4].reshape(-1, 2) * width
        
        # 提取颜色
        color = stroke[-4:]
        
        # 创建路径
        shapes.append(
            pydiffvg.Path(
                num_control_points=torch.LongTensor(num_control_points),
                points=torch.FloatTensor(path_points),
                stroke_width=torch.tensor(0.0),
                is_closed=True))
        
        # 创建形状组
        groups.append(
            pydiffvg.ShapeGroup(
                shape_ids=torch.LongTensor([num]),
                fill_color=torch.FloatTensor(color)))
    
    # 将路径和形状组添加到SVG对象
    svg_object.layers.append({'shapes': shapes, 'shape_groups': groups})
    svg_object.shape_id = num_strokes
    
    return svg_object


def _configure_diffvg_device(device: torch.device):
    """根据 torch.device 配置 pydiffvg 是否使用 GPU。"""  # 说明用途
    if device.type == 'cuda':  # 判断是否为 CUDA 设备
        pydiffvg.set_use_gpu(True)  # 开启 GPU
    else:  # 否则
        pydiffvg.set_use_gpu(False)  # 使用 CPU


def bitmap_to_svg(
    input_image_path: str,  # 输入位图路径（PNG/JPG）
    checkpoint_path: str,  # 模型检查点路径
    output_svg_path: Optional[str] = None,  # 输出 SVG 路径（可选，不传则自动生成默认路径）
    device: Union[str, torch.device] = "cpu",  # 推理设备（默认 CPU）
    width: int = 224,  # 输入与输出的边长
    stroke_num: int = 128,  # 笔画数量
    path_num: int = 4,  # 每个笔画的路径数量
    finetune_iter: int = 50,  # 微调迭代次数（0 则不微调）
    lr_path: float = 1.0,  # 路径点学习率
    lr_color: float = 0.01,  # 颜色学习率
    verbose: bool = True,  # 是否打印日志
):
    """将位图转换为 SVG，保存到文件，并返回 (svg_object, svg_path)。"""  # 对外说明
    if not os.path.exists(input_image_path):  # 检查输入文件
        raise FileNotFoundError(f"输入文件不存在: {input_image_path}")  # 抛错
    if not os.path.exists(checkpoint_path):  # 检查模型文件
        raise FileNotFoundError(f"模型检查点不存在: {checkpoint_path}")  # 抛错

    device = torch.device(device) if not isinstance(device, torch.device) else device  # 统一 device 类型
    if verbose:  # 如果需要日志
        print(f"使用设备: {device}")  # 打印设备

    _configure_diffvg_device(device)  # 配置 diffvg 设备

    if verbose:  # 如果需要日志
        print(f"正在加载图片: {input_image_path}")  # 打印提示
    img_tensor, img_np, original_size = load_image(input_image_path, width, device)  # 加载图片
    if verbose:  # 如果需要日志
        print(f"图片原始尺寸: {original_size}")  # 打印原始尺寸

    model = load_model(  # 加载模型
        checkpoint_path=checkpoint_path,  # 检查点路径
        device=device,  # 设备
        width=width,  # 宽度
        stroke_num=stroke_num,  # 笔画数
        path_num=path_num,  # 路径数
    )

    if verbose:  # 如果需要日志
        print("正在进行推理...")  # 打印提示
    with torch.no_grad():  # 关闭梯度
        if verbose:  # 如果需要日志
            print("正在预测笔画参数...")  # 打印提示
        strokes = model.predict_path(img_tensor)  # 预测笔画

    if verbose:  # 如果需要日志
        print("正在创建SVG对象...")  # 打印提示
    svg_object = strokes_to_svg_object(strokes, width, device, path_num)  # 转为 SVGObject

    if finetune_iter and finetune_iter > 0:  # 是否微调
        if verbose:  # 如果需要日志
            print(f"正在进行 {finetune_iter} 次微调迭代...")  # 打印提示
        svg_object.set_target(img_np)  # 设置目标图像
        svg_object.finetune(  # 微调
            lr_path=lr_path,  # 路径学习率
            lr_color=lr_color,  # 颜色学习率
            num_iter=finetune_iter,  # 迭代次数
            loss_type="mse",  # 损失
        )
        if verbose:  # 如果需要日志
            print("微调完成！")  # 打印完成

    if output_svg_path:  # 若指定输出路径
        output_path = Path(output_svg_path)  # 标准化
    else:  # 否则默认与输入同名同目录
        output_path = Path(input_image_path).with_suffix(".svg")  # 默认输出

    output_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在

    if verbose:  # 如果需要日志
        print("正在保存SVG文件...")  # 打印提示
    svg_object.save(str(output_path))  # 保存 SVG
    if verbose:  # 如果需要日志
        print(f"SVG文件已保存至: {str(output_path)}")  # 打印路径
        print("推理完成！")  # 打印结束

    return svg_object, str(output_path)  # 返回对象与路径


def main():
    """命令行入口（兼容原脚本用法）。"""  # CLI 说明
    args = parse_args()  # 解析参数
    bitmap_to_svg(  # 调用可复用接口
        input_image_path=args.input,  # 输入
        checkpoint_path=args.checkpoint,  # 检查点
        output_svg_path=args.output,  # 输出
        device=args.device,  # 设备
        width=args.width,  # 宽度
        stroke_num=args.stroke_num,  # 笔画
        path_num=args.path_num,  # 路径
        finetune_iter=args.finetune_iter,  # 微调
        lr_path=args.lr_path,  # 路径学习率
        lr_color=args.lr_color,  # 颜色学习率
        verbose=True,  # 打印日志
    )


if __name__ == '__main__':
    main()

