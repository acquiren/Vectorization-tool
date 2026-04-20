#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
编码器模块

包含特征提取器、笔画注意力头和粗粒度模型等组件
"""

import torch
import torch.nn as nn
from collections import OrderedDict
import timm 
from timm.models.vision_transformer import Block
from util.cross_attention import CrossAttentionBlock
from torchvision import transforms
from torchvision import models
from util.utils import SignWithSigmoidGrad


class StrokeAttentionHead(nn.Module):
    '''
    基于交叉注意力的笔画预测头
    '''
    def __init__(self, stroke_num=256, stroke_dim=13, encoder_embed_dim=768, self_attn_depth=4,control_num=False,num_loss=False):
        """
        初始化笔画注意力头
        
        Args:
            stroke_num: 笔画数量
            stroke_dim: 每个笔画的维度
            encoder_embed_dim: 编码器嵌入维度
            self_attn_depth: 自注意力深度
            control_num: 是否控制笔画数量
            num_loss: 是否使用数量损失
        """
        super(StrokeAttentionHead, self).__init__()
        self.control_num = control_num  # 是否控制笔画数量
        # 如果使用数量损失，增加一个维度
        if num_loss:
            stroke_dim += 1
        self.stroke_dim = stroke_dim  # 笔画维度
        # 如果控制笔画数量，添加控制数量的token
        if control_num:
            self.control_num_tokens = nn.Parameter(torch.zeros(1, 8, stroke_num))
        # 笔画token参数
        self.stroke_tokens = nn.Parameter(torch.zeros(1, stroke_dim, stroke_num))
        # 交叉注意力块
        self.cross_attn_block = CrossAttentionBlock(x_dim=encoder_embed_dim, y_dim=stroke_num, num_heads=8)
        # 自注意力块序列
        self.self_attn_blocks = nn.Sequential(*[
            Block(
                dim=stroke_num, num_heads=8, mlp_ratio=4., qkv_bias=True,
                attn_drop=0., norm_layer=nn.LayerNorm, act_layer=nn.GELU)
            for i in range(self_attn_depth)])
        # 线性层输出头
        self.linear_head = nn.Linear(stroke_dim, stroke_dim)
    
    def forward(self, x, num=None):
        """
        前向传播
        
        Args:
            x: 输入特征
            num: 笔画数量（仅当control_num=True时使用）
        
        Returns:
            torch.Tensor: 预测的笔画参数
        """
        if self.control_num:
            assert num is not None, 'num should not be None if control_num is True'
            # 复制控制数量token并乘以num
            control_num_tokens = self.control_num_tokens.repeat(x.shape[0], 1, 1) * num
            # 拼接笔画token和控制token
            x = self.cross_attn_block(x, torch.cat([self.stroke_tokens.repeat(x.shape[0], 1, 1), control_num_tokens], dim=1))[:, :self.stroke_dim, :]
            # 通过自注意力块
            x = self.self_attn_blocks(x)
            # 通过线性层并取前num个笔画
            x = self.linear_head(x.permute(0, 2, 1))[:, :num, :]
        else:
            # 通过交叉注意力块
            x = self.cross_attn_block(x, self.stroke_tokens.repeat(x.shape[0], 1, 1))
            # 通过自注意力块
            x = self.self_attn_blocks(x)
            # 通过线性层
            x = self.linear_head(x.permute(0, 2, 1))
        return x
    

class Coarse_model(nn.Module):
    '''
    基于交叉注意力的粗粒度预测器
    '''
    def __init__(self, stroke_num=512, stroke_dim=13, self_attn_depth=1,control_num=False,num_loss=False):
        """
        初始化粗粒度模型
        
        Args:
            stroke_num: 笔画数量
            stroke_dim: 笔画维度
            self_attn_depth: 自注意力深度
            control_num: 是否控制笔画数量
            num_loss: 是否使用数量损失
        """
        super(Coarse_model, self).__init__()
        # 使用ViT-Small作为特征提取器
        self.feature_extractor = timm.create_model('vit_small_patch16_224_dino', pretrained=False)
        # 创建笔画注意力头
        self.stroke_head = StrokeAttentionHead(stroke_num=stroke_num, stroke_dim=stroke_dim, encoder_embed_dim=self.feature_extractor.embed_dim, self_attn_depth=self_attn_depth, control_num=control_num, num_loss=num_loss)
        # 224x224的大小调整
        self.resize_224 = transforms.Resize((224, 224))
    
    def extract_features(self, x):
        """
        提取特征
        
        Args:
            x: 输入图像
        
        Returns:
            torch.Tensor: 提取的特征
        """
        # 补丁嵌入
        x = self.feature_extractor.patch_embed(x)
        # 分类token
        cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)
        # 拼接分类token和补丁嵌入
        x = torch.cat((cls_token, x), dim=1)
        # 添加位置嵌入
        x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
        # 通过Transformer块
        x = self.feature_extractor.blocks(x)
        # 归一化
        x = self.feature_extractor.norm(x)
        return x[:, 1:]  # 去掉分类token

    def forward(self, x, num=None, **kwargs):
        """
        前向传播
        
        Args:
            x: 输入图像
            num: 笔画数量（可选）
            **kwargs: 其他参数
        
        Returns:
            torch.Tensor: 预测的笔画参数
        """
        # 如果输入大小不是224x224，调整大小
        if x.size(-1) != 224:
            x = self.resize_224(x)
        # 提取特征
        x = self.extract_features(x)
        # 通过笔画注意力头
        x = self.stroke_head(x, num)
        # 使用sigmoid激活
        x = torch.sigmoid(x)
        # 如果维度是28，对最后一个维度应用特殊处理
        if x.size(-1) == 28:
            x = torch.cat([x[:, :, :27], SignWithSigmoidGrad.apply(x[:, :, 27:28] - 0.5)], dim=-1)
        return x


class path_predictor(nn.Module):
    def __init__(self, stage, stroke_num=512, stroke_dim=13, self_attn_depth=1, control_num=False, use_resnet=False, num_loss=False):
        """
        初始化路径预测器
        
        Args:
            stage: 阶段（0或1）
            stroke_num: 笔画数量
            stroke_dim: 笔画维度
            self_attn_depth: 自注意力深度
            control_num: 是否控制笔画数量
            use_resnet: 是否使用ResNet
            num_loss: 是否使用数量损失
        """
        super(path_predictor, self).__init__()

        self.stage = stage  # 阶段
        self.stroke_num = stroke_num  # 笔画数量
        self.resize_224 = transforms.Resize((224, 224))  # 224x224大小调整
        if stage == 0:
            # 阶段0：使用ViT作为特征提取器
            self.feature_extractor = timm.create_model('vit_small_patch16_224_dino', pretrained=False)
            self.stroke_head = StrokeAttentionHead(stroke_num=stroke_num, stroke_dim=stroke_dim,
                                                   encoder_embed_dim=self.feature_extractor.embed_dim,
                                                   self_attn_depth=self_attn_depth, control_num=control_num, num_loss=num_loss)
        else:
            # 阶段1：使用映射层和可选的ResNet
            self.use_resnet = use_resnet
            # 输入映射层
            self.map_in = nn.Sequential(
                nn.Conv2d(6, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 3, 3, 1, 1),
                nn.BatchNorm2d(3),
                nn.ReLU(),
            )
            if not use_resnet:
                # 不使用ResNet，使用ViT
                self.feature_extractor = timm.create_model('vit_small_patch16_224_dino', pretrained=False)
                self.stroke_head = StrokeAttentionHead(stroke_num=128, stroke_dim=stroke_dim,
                                                       encoder_embed_dim=self.feature_extractor.embed_dim,
                                                       self_attn_depth=self_attn_depth, control_num=control_num)
                self.map_out = nn.Linear(128, stroke_num)
            else:
                # 使用ResNet
                self.resnet = models.resnet50(pretrained=True)
                num_ftrs = self.resnet.fc.in_features
                print('num featrues', num_ftrs)
                self.resnet.fc = nn.Linear(num_ftrs, stroke_num * stroke_dim)
    
    def extract_features(self, x):
        """
        提取特征
        
        Args:
            x: 输入图像
        
        Returns:
            torch.Tensor: 提取的特征
        """
        # 补丁嵌入
        x = self.feature_extractor.patch_embed(x)
        # 分类token
        cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)
        # 拼接分类token和补丁嵌入
        x = torch.cat((cls_token, x), dim=1)
        # 添加位置嵌入
        x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
        # 通过Transformer块
        x = self.feature_extractor.blocks(x)
        # 归一化
        x = self.feature_extractor.norm(x)
        return x[:, 1:]  # 去掉分类token

    def forward(self, x, canvas):
        """
        前向传播
        
        Args:
            x: 输入图像
            canvas: 画布图像
        
        Returns:
            torch.Tensor: 预测的笔画参数
        """
        # 调整输入大小
        if x.size(-1) != 224:
            x = self.resize_224(x)
            if canvas is not None:
                canvas = self.resize_224(canvas)
        if self.stage == 0:
            # 阶段0：直接使用ViT
            x = self.extract_features(x)
            x = self.stroke_head(x)
        else:
            # 阶段1：结合输入和画布
            if self.use_resnet:
                # 使用ResNet
                x = torch.cat([x, canvas], dim=1)
                x = self.map_in(x)
                x = self.resnet(x)
                x = x.view(x.size(0), self.stroke_num, -1)
            else:
                # 使用ViT
                x = torch.cat([x, canvas], dim=1)
                x = self.map_in(x)
                x = self.extract_features(x)
                x = self.stroke_head(x)
                x = self.map_out(x.permute(0, 2, 1)).permute(0, 2, 1)

        # 使用sigmoid激活
        x = torch.sigmoid(x)
        # 如果维度是28，对最后一个维度应用特殊处理
        if x.size(-1) == 28:
            x = torch.cat([x[:, :, :27], SignWithSigmoidGrad.apply(x[:, :, 27:28] - 0.5)], dim=-1)
        return x


class Refinement_model(nn.Module):
    def __init__(self, stroke_num=512, stroke_dim=13, self_attn_depth=1, control_num=False, use_resnet=False):
        """
        初始化精修模型
        
        Args:
            stroke_num: 笔画数量
            stroke_dim: 笔画维度
            self_attn_depth: 自注意力深度
            control_num: 是否控制笔画数量
            use_resnet: 是否使用ResNet
        """
        super(Refinement_model, self).__init__()
        # 阶段0的编码器
        self.encoder1 = path_predictor(0, stroke_num, stroke_dim, self_attn_depth, control_num=control_num,
                                    use_resnet=use_resnet, num_loss=True)
        # 阶段1的编码器
        self.encoder2 = path_predictor(1, 8, stroke_dim, self_attn_depth, control_num=control_num, use_resnet=use_resnet)

    def forward(self, x, canvas, step=0):
        """
        前向传播
        
        Args:
            x: 输入图像
            canvas: 画布图像
            step: 步骤（0或1）
        
        Returns:
            torch.Tensor: 预测的笔画参数
        """
        if step == 0:
            return self.encoder1(x, canvas)
        else:
            return self.encoder2(x, canvas)

