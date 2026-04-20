#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态时间规整（DPW）损失模块

用于计算两个笔画序列之间的距离
"""

import torch
import numpy as np


def euclidean_dist_func(x, y):
    """
    计算欧氏距离
    
    Args:
        x: 第一个张量，形状为 [B, N, D]
        y: 第二个张量，形状为 [B, M, D]
    
    Returns:
        torch.Tensor: 距离矩阵，形状为 [B, N, M]
    """
    n = x.size(1)  # x的序列长度
    m = y.size(1)  # y的序列长度
    d = x.size(2)  # 特征维度
    # 扩展维度以便广播
    x = x.unsqueeze(2).expand(-1, n, m, d)
    y = y.unsqueeze(1).expand(-1, n, m, d)
    # 计算欧氏距离的平方
    return torch.pow(x - y, 2).sum(3)


def loss_dpw(strokes, strokes_b, gamma=0.01, bandwidth=100):
    """
    计算DPW损失
    
    Args:
        strokes: 生成的路径，形状为 [B, N, D]
        strokes_b: 目标路径，形状为 [B, M, D]
        gamma: 温度参数
        bandwidth: 带宽限制
    
    Returns:
        torch.Tensor: DPW损失值
    """
    # 计算距离矩阵
    d_xy = euclidean_dist_func(strokes, strokes_b)
    d_xy.retain_grad()  # 保留梯度用于反向传播
    
    B = d_xy.shape[0]  # 批次大小
    N = d_xy.shape[1]  # 第一个序列的长度
    M = d_xy.shape[2]  # 第二个序列的长度
    
    # 初始化动态规划表
    p = torch.ones((d_xy.shape[0], d_xy.shape[1] + 2, d_xy.shape[2] + 2), requires_grad=True)
    p = p * torch.inf  # 初始化为无穷大
    p[:, 0, :] = 0  # 第一行初始化为0
    
    q = torch.ones((d_xy.shape[0], d_xy.shape[1] + 2, d_xy.shape[2] + 2), requires_grad=True)
    q = q * torch.inf  # 初始化为无穷大
    
    p.retain_grad()
    q.retain_grad()
    
    # 初始化结果
    result = torch.ones((d_xy.shape[0]), requires_grad=True)
    result = result * torch.inf
    result.retain_grad()
    
    P = p.clone()
    Q = q.clone()
    
    # 动态规划过程
    for b in range(B):
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                # 检查带宽限制
                if 0 < bandwidth < np.abs(i - j):
                    continue
                # 计算softmin
                r0 = -Q[b, i - 1, j] / gamma
                r1 = -P[b, i - 1, j] / gamma
                r2 = -Q[b, i, j - 1] / gamma
                r3 = -P[b, i, j - 1] / gamma
                rmax = max(r0, r1)
                rsum = torch.exp(r0 - rmax) + torch.exp(r1 - rmax)
                softmin = -gamma * (torch.log(rsum) + rmax)
                P[b, i, j] = d_xy[b, i - 1, j - 1] + softmin
                
                rmax0 = max(r2, r3)
                rsum0 = torch.exp(r2 - rmax0) + torch.exp(r3 - rmax0)
                softmin0 = -gamma * (torch.log(rsum0) + rmax0)
                Q[b, i, j] = softmin0
                if r2 == -torch.inf and r3 == -torch.inf:
                    Q[b, i, j] = torch.inf
        # 计算最终结果
        p0 = -Q[b, N, M] / gamma
        q0 = -P[b, N, M] / gamma
        rmax1 = max(p0, q0)
        rsum1 = torch.exp(p0 - rmax1) + torch.exp(q0 - rmax1)
        softmin1 = -gamma * (torch.log(rsum1) + rmax1)
        result[b] = softmin1

    return result.mean()  # 返回批次的平均损失

