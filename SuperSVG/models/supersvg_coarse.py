#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperSVG coarse阶段模型

包含完整的SuperSVG coarse阶段模型，用于将图像转换为SVG矢量图
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Coarse_model
import pydiffvg
import lpips
from util import SVR_render
import random
from util import mophology

# 全局变量
channel_mean = torch.tensor([0.485, 0.456, 0.406])  # ImageNet均值
channel_std = torch.tensor([0.229, 0.224, 0.225])  # ImageNet标准差
pydiffvg.set_print_timing(False)  # 不打印diffvg的计时信息
pydiffvg.set_use_gpu(True)  # 使用GPU
MEAN = [-mean / std for mean, std in zip(channel_mean, channel_std)]  # 反归一化均值
STD = [1 / std for std in channel_std]  # 反归一化标准差
torch.multiprocessing.set_start_method('spawn', force=True)  # 设置多进程启动方式


class SuperSVG_coarse(nn.Module):
    """
    SuperSVG coarse阶段模型
    
    用于将图像转换为SVG矢量图的粗粒度模型
    """
    def __init__(self, stroke_num=128, path_num=4, width=128, control_num=False, self_attn_depth=1, num_loss=False):
        """
        初始化SuperSVG coarse阶段模型
        
        Args:
            stroke_num: 笔画数量
            path_num: 每个笔画的路径数量
            width: 画布宽度
            control_num: 是否控制笔画数量
            self_attn_depth: 自注意力深度
            num_loss: 是否使用数量损失
        """
        super(SuperSVG_coarse, self).__init__()
        self.control_num = control_num  # 是否控制笔画数量
        self.path_num = path_num  # 每个笔画的路径数量
        # 创建编码器
        self.encoder = Coarse_model(stroke_num=stroke_num, stroke_dim=path_num * 6 + 3, control_num=control_num, self_attn_depth=self_attn_depth, num_loss=num_loss)
        self.stroke_num = stroke_num  # 笔画数量
        self.device = 'cuda'  # 设备
        self.render = SVR_render.SVGObject(size=(width, width))  # SVG渲染对象
        self.width = width  # 画布宽度
        self.loss_fn_vgg = None  # VGG损失函数（暂未使用）

    def forward(self, x, mask=None, num=None, **kwargs):
        """
        前向传播
        
        Args:
            x: 输入图像
            mask: 掩码（可选）
            num: 笔画数量（可选）
            **kwargs: 其他参数
        
        Returns:
            tuple: (pred, x) 预测图像和输入图像
        """
        # 如果有掩码，应用掩码
        if mask is not None:
            x = x * mask - (1 - mask)
        # 通过编码器预测笔画
        if self.control_num:
            if num is None:
                num = random.randint(1, 64)
            strokes = self.encoder(x, num)
        else:
            strokes = self.encoder(x)

        # 渲染SVG
        pred = self.rendering(strokes, **kwargs)

        return pred, x
    
    def predict_path(self, x, num=None, **kwargs):
        """
        预测路径参数
        
        Args:
            x: 输入图像
            num: 笔画数量（可选）
            **kwargs: 其他参数
        
        Returns:
            torch.Tensor: 预测的笔画参数
        """
        # 通过编码器预测笔画
        if self.control_num:
            if num is None:
                num = random.randint(1, 64)
            strokes = self.encoder(x, num)
        else:
            strokes = self.encoder(x)

        return strokes

    def rendering(self, strokes, save_svg_path=None):
        """
        渲染SVG
        
        Args:
            strokes: 笔画参数
            save_svg_path: 保存SVG的路径（可选）
        
        Returns:
            torch.Tensor: 渲染的图像
        """
        imgs = []
        # 如果笔画维度是27，添加一个维度
        if strokes.size(-1) == 27:
            strokes = torch.cat([strokes, torch.ones(strokes.size(0), strokes.size(1), 1).to(strokes.device)], dim=2)
        strokes = strokes.float()
        # 每个路径的控制点数量
        num_control_points = [2] * self.path_num
        # 遍历每个批次
        for b in range(strokes.size(0)):
            shapes = []
            groups = []
            # 遍历每个笔画
            for num in range(strokes.size(1)):
                # 创建路径
                shapes.append(
                    pydiffvg.Path(
                        num_control_points=torch.LongTensor(num_control_points),
                        points=strokes[b][num][:-4].reshape(-1, 2) * self.width,
                        stroke_width=torch.tensor(0.0),
                        is_closed=True))
                # 创建形状组
                groups.append(
                    pydiffvg.ShapeGroup(
                        shape_ids=torch.LongTensor([num]),
                        fill_color=strokes[b][num][-4:]))
            # 序列化场景
            scene_args = pydiffvg.RenderFunction.serialize_scene(self.width, self.width, shapes, groups)
            _render = pydiffvg.RenderFunction.apply
            # 渲染
            img = _render(self.width, self.width, 2, 2, 0, None, *scene_args)
            imgs.append(img.permute(2, 0, 1))
        # 堆叠图像
        imgs = torch.stack(imgs, dim=0)
        # 保存SVG
        if save_svg_path is not None:
            pydiffvg.save_svg(save_svg_path, self.width, self.width, shapes, groups)
        return imgs

    def loss(self, x, mask=None, epoch_id=0, num_loss=True):
        """
        计算损失
        
        Args:
            x: 输入图像
            mask: 掩码
            epoch_id: epoch的ID
            num_loss: 是否使用数量损失
        
        Returns:
            tuple: (loss, log_loss) 总损失和损失字典
        """
        # 通过编码器预测笔画
        strokes = self.encoder(x * mask - (1 - mask))
        # 渲染预测图像
        pred = self.rendering(strokes)[:, :3, :, :]
        # 计算MSE损失
        loss_mse = F.mse_loss(pred * mask, x * mask) / (mask.sum() / (mask.size(0) * mask.size(-1) * mask.size(-2)))
        # 如果笔画维度是27，添加透明度
        if strokes.size(-1) == 27:
            new_strokes = torch.cat([strokes[:, :, :-3], torch.ones_like(strokes[:, :, -3:].to(strokes.device))], dim=-1)
        else:
            new_strokes = torch.cat([strokes[:, :, :-4], torch.ones_like(strokes[:, :, -4:].to(strokes.device))],
                                    dim=-1)
        # 渲染新的图像
        pred = self.rendering(new_strokes)[:, :1, :, :]
        pred.reshape(x.shape[0], -1, pred.shape[-2], pred.shape[-1])
        # 计算掩码损失的权重
        lambda_mask = max(0.05 - 0.005 * (epoch_id + 1), 0.01)
        # 膨胀掩码
        mask = mophology.dilation(mask, m=2)
        # 计算掩码损失
        loss_mask = ((pred * (1 - mask)).sum()) / ((1 - mask).sum()) * lambda_mask
        # 总损失
        loss = loss_mse + loss_mask
        log_loss = {}
        log_loss['loss_pixel'] = loss_mse
        log_loss['loss_mask'] = loss_mask.item()
        # 如果使用数量损失且笔画维度是28
        if num_loss and strokes.size(-1) == 28:
            loss_num = strokes[:, :, -1].sum(dim=-1)
            loss_num = loss_num.mean()
            loss += loss_num * 0.00001
            log_loss['loss_num'] = loss_num * 0.00001
            log_loss['path_num'] = loss_num
        log_loss["loss"] = loss.item()
        return loss, log_loss

