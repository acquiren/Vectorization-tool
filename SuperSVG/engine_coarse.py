#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练引擎模块（coarse阶段）

使用方法：
1. 导入 train_one_epoch 和 evaluate 函数
2. 调用 train_one_epoch 进行一个epoch的训练
3. 调用 evaluate 进行模型评估
"""

import math
import os
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from torchvision.utils import save_image
import time


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None,
                    epoch_id=0,
                    wandb=None,
                    scheduler=None):
    """
    训练一个epoch
    
    Args:
        model: 待训练的模型
        data_loader: 数据加载器
        optimizer: 优化器
        device: 设备（CPU或GPU）
        epoch: 当前epoch编号
        loss_scaler: 损失缩放器（用于混合精度训练）
        log_writer: TensorBoard日志记录器
        args: 命令行参数
        epoch_id: epoch的ID标识
        wandb: Weights & Biases记录器
        scheduler: 学习率调度器
    
    Returns:
        dict: 包含所有指标的字典
    """
    # 设置模型为训练模式
    model.train(True)
    # 初始化指标记录器
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20  # 每20步打印一次日志

    # 梯度累积的迭代次数
    accum_iter = args.accum_iter

    # 清零梯度
    optimizer.zero_grad()

    # 打印日志目录
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    # 遍历数据加载器
    for data_iter_step, (imgs,masks) in enumerate(metric_logger.log_every(data_loader, print_freq, header,wandb)):
        # 移除批次维度的第一个维度
        imgs=imgs.squeeze(0)
        masks=masks.squeeze(0)
        # 将数据移动到指定设备
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        # 调整学习率
        if scheduler is None:
            if data_iter_step % accum_iter == 0:
                lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        else:
            scheduler.step(data_iter_step / len(data_loader) + epoch)

        # 前向传播（使用混合精度）
        with torch.cuda.amp.autocast():
            if args.distributed:
                # 分布式训练模式
                loss,kwargs=model.module.loss(imgs,mask=masks,epoch_id=epoch_id)
            else:
                # 单GPU训练模式
                loss,kwargs = model.loss(imgs, mask=masks, epoch_id=epoch_id)
        # 更新指标记录器
        metric_logger.update(**kwargs)

        # 获取损失值
        loss_value = loss.item()

        # 检查损失是否为有限值
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        # 梯度累积（除以累积次数）
        loss /= accum_iter
        
        t1 = time.time()
        
        # 反向传播和参数更新
        if args.distributed:
            loss_scaler(loss, optimizer, parameters=model.module.encoder.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
        else:
            loss_scaler(loss, optimizer, parameters=model.encoder.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        # 每累积次数清零一次梯度
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        # 同步CUDA操作
        torch.cuda.synchronize()

        # 获取当前学习率并更新指标
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # 对损失进行多进程规约
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        # 记录到TensorBoard
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # 评估模型
    evaluate(model, imgs, masks, log_writer, args, epoch_id,wandb=wandb)
    # 同步所有进程的指标
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # 返回所有指标的全局平均值
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, images,masks, log_writer, args, epoch_id,wandb):
    """
    评估模型
    
    Args:
        model: 待评估的模型
        images: 输入图像
        masks: 输入掩码
        log_writer: TensorBoard日志记录器
        args: 命令行参数
        epoch_id: epoch的ID标识
        wandb: Weights & Biases记录器
    
    Returns:
        None
    """
    # 设置模型为评估模式
    model.eval()
    # 前向传播（使用混合精度，不计算梯度）
    with torch.cuda.amp.autocast():
        output, gt = model(images,masks,num=128)
    # 创建保存图像的目录
    os.makedirs(args.log_dir + '/imgs', exist_ok=True)
    # 保存输出图像和真实图像
    save_image(torch.cat([output[:8, :3, :, :], gt[:8]], dim=0), args.log_dir + '/imgs/%d.jpg' % epoch_id, nrow=8,
               normalize=False)

