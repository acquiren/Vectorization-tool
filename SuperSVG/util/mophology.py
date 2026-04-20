#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
形态学操作模块

包含腐蚀和膨胀等形态学操作的实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Erosion2d(nn.Module):
    """
    2D腐蚀操作模块
    """
    def __init__(self, m=1):
        """
        初始化腐蚀模块
        
        Args:
            m: 腐蚀核的大小
        """
        super(Erosion2d, self).__init__()
        self.m = m  # 腐蚀核大小
        self.pad = [m, m, m, m]  # 填充大小
        self.unfold = nn.Unfold(2 * m + 1, padding=0, stride=1)  # 展开操作

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量
        
        Returns:
            torch.Tensor: 腐蚀后的张量
        """
        batch_size, c, h, w = x.shape
        # 填充输入
        x_pad = F.pad(x, pad=self.pad, mode='constant', value=1e9)
        # 展开
        channel = self.unfold(x_pad).view(batch_size, c, -1, h, w)
        # 取最小值（腐蚀操作）
        result = torch.min(channel, dim=2)[0]
        return result


def erosion(x, m=1):
    """
    腐蚀函数（函数式）
    
    Args:
        x: 输入张量
        m: 腐蚀核的大小
    
    Returns:
        torch.Tensor: 腐蚀后的张量
    """
    b, c, h, w = x.shape
    # 填充输入
    x_pad = F.pad(x, pad=[m, m, m, m], mode='constant', value=1e9)
    # 展开
    channel = nn.functional.unfold(x_pad, 2 * m + 1, padding=0, stride=1).view(b, c, -1, h, w)
    # 取最小值（腐蚀操作）
    result = torch.min(channel, dim=2)[0]
    return result


class Dilation2d(nn.Module):
    """
    2D膨胀操作模块
    """
    def __init__(self, m=1):
        """
        初始化膨胀模块
        
        Args:
            m: 膨胀核的大小
        """
        super(Dilation2d, self).__init__()
        self.m = m  # 膨胀核大小
        self.pad = [m, m, m, m]  # 填充大小
        self.unfold = nn.Unfold(2 * m + 1, padding=0, stride=1)  # 展开操作

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量
        
        Returns:
            torch.Tensor: 膨胀后的张量
        """
        batch_size, c, h, w = x.shape
        # 填充输入
        x_pad = F.pad(x, pad=self.pad, mode='constant', value=-1e9)
        # 展开
        channel = self.unfold(x_pad).view(batch_size, c, -1, h, w)
        # 取最大值（膨胀操作）
        result = torch.max(channel, dim=2)[0]
        return result


def dilation(x, m=1):
    """
    膨胀函数（函数式）
    
    Args:
        x: 输入张量
        m: 膨胀核的大小
    
    Returns:
        torch.Tensor: 膨胀后的张量
    """
    b, c, h, w = x.shape
    # 填充输入
    x_pad = F.pad(x, pad=[m, m, m, m], mode='constant', value=-1e9)
    # 展开
    channel = nn.functional.unfold(x_pad, 2 * m + 1, padding=0, stride=1).view(b, c, -1, h, w)
    # 取最大值（膨胀操作）
    result = torch.max(channel, dim=2)[0]
    return result

