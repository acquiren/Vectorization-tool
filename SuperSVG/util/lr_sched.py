#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学习率调度模块

包含cosine学习率调度函数
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math


def adjust_learning_rate(optimizer, epoch, args):
    """
    调整学习率
    
    使用带有warmup的cosine衰减调度
    
    Args:
        optimizer: 优化器
        epoch: 当前epoch
        args: 命令行参数
    """
    if epoch < args.warmup_epochs:
        # warmup阶段：线性增加学习率
        lr = args.lr * epoch / args.warmup_epochs
    else:
        # cosine衰减阶段
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    # 更新优化器的学习率
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

