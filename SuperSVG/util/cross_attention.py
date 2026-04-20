#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交叉注意力模块

包含交叉注意力层和交叉注意力块的实现
"""

import torch
import torch.hub
import torch.nn as nn
from itertools import repeat
import collections.abc


# 从PyTorch内部代码复制的函数
def _ntuple(n):
    """
    创建一个将输入转换为n元组的函数
    
    Args:
        n: 元组的长度
    
    Returns:
        function: 转换函数
    """
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    """
    确保值是可整除的
    
    Args:
        v: 输入值
        divisor: 除数
        min_value: 最小值
        round_limit: 舍入限制
    
    Returns:
        int: 可整除的值
    """
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # 确保向下取整不会超过10%
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class Mlp(nn.Module):
    """
    多层感知机（MLP）模块
    
    用于Vision Transformer、MLP-Mixer等网络中
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """
        初始化MLP模块
        
        Args:
            in_features: 输入特征维度
            hidden_features: 隐藏层特征维度
            out_features: 输出特征维度
            act_layer: 激活函数
            drop: dropout概率
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量
        
        Returns:
            torch.Tensor: 输出张量
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """
    随机丢弃路径（Stochastic Depth）
    
    用于残差块的主路径中
    
    Args:
        x: 输入张量
        drop_prob: 丢弃概率
        training: 是否训练模式
        scale_by_keep: 是否按保留概率缩放
    
    Returns:
        torch.Tensor: 输出张量
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 处理不同维度的张量
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """
    随机丢弃路径（Stochastic Depth）模块
    
    用于残差块的主路径中
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        """
        初始化DropPath模块
        
        Args:
            drop_prob: 丢弃概率
            scale_by_keep: 是否按保留概率缩放
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量
        
        Returns:
            torch.Tensor: 输出张量
        """
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class CrossAttention(nn.Module):
    """
    交叉注意力层
    """
    def __init__(
            self,
            x_dim,
            y_dim,
            num_heads=8,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.,
    ):
        """
        初始化交叉注意力层
        
        Args:
            x_dim: x的维度
            y_dim: y的维度
            num_heads: 注意力头的数量
            qkv_bias: 是否使用偏置
            attn_drop: 注意力dropout概率
            proj_drop: 投影dropout概率
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = x_dim // num_heads
        # 缩放因子
        self.scale = head_dim ** -0.5

        self.wq = nn.Linear(y_dim, y_dim, bias=qkv_bias)
        self.wk = nn.Linear(x_dim, y_dim, bias=qkv_bias)
        self.wv = nn.Linear(x_dim, y_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(y_dim, y_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        """
        前向传播
        
        Args:
            x: 第一个输入张量
            y: 第二个输入张量
        
        Returns:
            torch.Tensor: 输出张量
        """
        B, N, C = x.shape
        B, Ny, Cy = y.shape
        # 计算Q、K、V
        q = self.wq(y).reshape(B, Ny, self.num_heads, Cy // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(x).reshape(B, N, self.num_heads, Cy // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(x).reshape(B, N, self.num_heads, Cy // self.num_heads).permute(0, 2, 1, 3)

        # 计算注意力权重
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 计算输出
        y = (attn @ v).transpose(1, 2).reshape(B, Ny, Cy)
        y = self.proj(y)
        y = self.proj_drop(y)
        return y


class CrossAttentionBlock(nn.Module):
    """
    交叉注意力块
    
    包含交叉注意力层和MLP层
    """
    def __init__(
            self,
            x_dim,
            y_dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        """
        初始化交叉注意力块
        
        Args:
            x_dim: x的维度
            y_dim: y的维度
            num_heads: 注意力头的数量
            mlp_ratio: MLP隐藏层比例
            qkv_bias: 是否使用偏置
            drop: dropout概率
            proj_drop: 投影dropout概率
            attn_drop: 注意力dropout概率
            drop_path: 路径丢弃概率
            act_layer: 激活函数
            norm_layer: 归一化层
        """
        super().__init__()
        self.norm1 = norm_layer(y_dim)
        self.attn = CrossAttention(
            x_dim,
            y_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        # 注意：这里的drop path用于随机深度，我们看看是否比dropout更好
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(y_dim)
        mlp_hidden_dim = int(y_dim * mlp_ratio)
        self.mlp = Mlp(in_features=y_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x, y):
        """
        前向传播
        
        Args:
            x: 第一个输入张量
            y: 第二个输入张量
        
        Returns:
            torch.Tensor: 输出张量
        """
        y = y + self.drop_path(self.attn(x, self.norm1(y)))
        y = y + self.drop_path(self.mlp(self.norm2(y)))
        return y

