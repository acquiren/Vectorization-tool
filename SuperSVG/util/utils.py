#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具函数模块

包含一些通用的工具函数和类
"""

import torch


class SignWithSigmoidGrad(torch.autograd.Function):
    """
    带Sigmoid梯度的符号函数
    
    前向传播使用符号函数，反向传播使用Sigmoid的梯度
    """
    @staticmethod
    def forward(ctx, x):
        """
        前向传播
        
        Args:
            x: 输入张量
        
        Returns:
            torch.Tensor: 符号函数的输出
        """
        result = (x > 0).float()  # 符号函数：x>0时为1，否则为0
        sigmoid_result = torch.sigmoid(x)  # 计算sigmoid用于反向传播
        ctx.save_for_backward(sigmoid_result)  # 保存sigmoid结果用于反向传播
        return result

    @staticmethod
    def backward(ctx, grad_result):
        """
        反向传播
        
        Args:
            grad_result: 损失对输出的梯度
        
        Returns:
            torch.Tensor: 损失对输入的梯度
        """
        (sigmoid_result,) = ctx.saved_tensors  # 获取保存的sigmoid结果
        if ctx.needs_input_grad[0]:
            # 使用sigmoid的梯度作为反向传播的梯度
            grad_input = grad_result * sigmoid_result * (1 - sigmoid_result)
        else:
            grad_input = None
        return grad_input

