#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学习率衰减模块

包含层衰减学习率设置函数
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    层衰减学习率参数组设置
    
    不同层使用不同的学习率，后几层使用更高的学习率
    
    Args:
        model: 模型
        weight_decay: 权重衰减
        no_weight_decay_list: 不使用权重衰减的参数列表
        layer_decay: 层衰减系数
    
    Returns:
        list: 参数组列表
    """
    param_group_names = {}  # 参数组名称字典
    param_groups = {}  # 参数组字典

    num_layers = len(model.blocks) + 1  # 总层数（包括patch_embed）

    # 计算缩放系数
    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # 跳过不需要梯度的参数

        # 不使用权重衰减的参数
        if n.endswith('cls_token') or n.endswith('pos_embed') or n.endswith('mask_token') or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        # 确定参数所在的层
        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        # 如果参数组不存在，则创建
        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        # 将参数添加到对应组
        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    print("parameter groups: \n%s" % param_group_names)
    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    为ViT参数分配层ID
    
    Args:
        name: 参数名称
        num_layers: 总层数
    
    Returns:
        int: 层ID
    """
    if name in ['cls_token', 'pos_embed'] or name.startswith('mask_token'):
        return 0  # 嵌入层
    elif name.startswith('patch_embed'):
        return 0  # patch嵌入层
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1  # 层ID
    else:
        return num_layers  # 输出层

