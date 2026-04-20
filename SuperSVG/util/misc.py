#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用工具模块

包含分布式训练、指标记录、模型保存等通用工具函数
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.distributed as dist
from torch._six import inf


class SmoothedValue(object):
    """
    平滑值跟踪器
    
    跟踪一系列值，并提供窗口内或全局的平滑值访问
    """

    def __init__(self, window_size=20, fmt=None):
        """
        初始化平滑值跟踪器
        
        Args:
            window_size: 窗口大小
            fmt: 格式化字符串
        """
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)  # 双端队列，用于存储最近的值
        self.total = 0.0  # 总值
        self.count = 0  # 总计数
        self.fmt = fmt  # 格式化字符串

    def update(self, value, n=1):
        """
        更新值
        
        Args:
            value: 新值
            n: 数量（默认为1）
        """
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        在进程间同步
        
        警告：不同步双端队列！
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        """获取中位数"""
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        """获取窗口平均值"""
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        """获取全局平均值"""
        return self.total / self.count

    @property
    def max(self):
        """获取最大值"""
        return max(self.deque)

    @property
    def value(self):
        """获取最新值"""
        return self.deque[-1]

    def __str__(self):
        """字符串表示"""
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    """
    指标记录器
    
    用于记录和打印训练过程中的各种指标
    """
    def __init__(self, delimiter="\t"):
        """
        初始化指标记录器
        
        Args:
            delimiter: 分隔符
        """
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        """
        更新指标
        
        Args:
            **kwargs: 指标名称和值的键值对
        """
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        """
        获取属性
        
        Args:
            attr: 属性名
        
        Returns:
            对应的值
        """
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        """字符串表示"""
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """在进程间同步"""
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """
        添加指标
        
        Args:
            name: 指标名称
            meter: SmoothedValue对象
        """
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, wandb=None):
        """
        遍历可迭代对象并记录日志
        
        Args:
            iterable: 可迭代对象
            print_freq: 打印频率
            header: 头部信息
            wandb: Weights & Biases记录器
        
        Yields:
            可迭代对象的元素
        """
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                    if wandb is not None:
                        for name, meter in self.meters.items():
                            tmp = str(meter)
                            if '(' in tmp:
                                wandb.log({name: float(tmp[tmp.index('(')+1:tmp.index(')')])})
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    为分布式训练设置打印
    
    当不是主进程时禁用打印
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # 打印时间戳
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    """
    检查分布式训练是否可用并已初始化
    
    Returns:
        bool: 是否可用并已初始化
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """
    获取世界大小（进程总数）
    
    Returns:
        int: 世界大小
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    获取当前进程的排名
    
    Returns:
        int: 排名
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """
    检查是否是主进程
    
    Returns:
        bool: 是否是主进程
    """
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """
    仅在主进程上保存
    
    Args:
        *args: 传递给torch.save的参数
        **kwargs: 传递给torch.save的关键字参数
    """
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    """
    初始化分布式训练模式
    
    Args:
        args: 命令行参数
    """
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    """
    带梯度范数计数的本地缩放器
    
    用于混合精度训练
    """
    state_dict_key = "amp_scaler"

    def __init__(self):
        """初始化缩放器"""
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        """
        调用缩放器
        
        Args:
            loss: 损失
            optimizer: 优化器
            clip_grad: 梯度裁剪值
            parameters: 参数列表
            create_graph: 是否创建计算图
            update_grad: 是否更新梯度
        
        Returns:
            torch.Tensor: 梯度范数
        """
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # 原位取消缩放优化器的梯度
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        """获取状态字典"""
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        """加载状态字典"""
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    """
    获取梯度范数
    
    Args:
        parameters: 参数列表
        norm_type: 范数类型
    
    Returns:
        torch.Tensor: 梯度范数
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    """
    保存模型
    
    Args:
        args: 命令行参数
        epoch: 当前epoch
        model: 模型
        model_without_ddp: 不包含DDP的模型
        optimizer: 优化器
        loss_scaler: 损失缩放器
    """
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    """
    加载模型
    
    Args:
        args: 命令行参数
        model_without_ddp: 不包含DDP的模型
        optimizer: 优化器
        loss_scaler: 损失缩放器
    """
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def all_reduce_mean(x):
    """
    对所有进程的x进行规约并求平均
    
    Args:
        x: 输入值
    
    Returns:
        float: 所有进程的平均值
    """
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x

