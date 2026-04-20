#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
彩色图像矢量描摹工具（IDE 运行版）

使用方法：
1. 在 IDE 中打开此脚本
2. 通过命令行参数或调用 main() 函数运行
3. 支持将位图转换为彩色 SVG 矢量图

功能说明：
- 使用 potrace 进行矢量描摹
- 支持多种颜色量化算法
- 支持多进程并行处理
- 支持自定义调色板重映射

基于 color_trace_multi 重构
原作者: ukurereh (2012)
Python3.8 重写: 赵豪杰 (2021)
代码规范化: 2024
"""

# =============================================================================
# 导入模块
# =============================================================================

import os
import sys
import shutil
import subprocess
import argparse
from glob import iglob
import functools
import multiprocessing
import queue
import tempfile
import time
import shlex
import re
from pprint import pprint

# -------------------------------------------------------------------------
# 解决“作为脚本运行 / 被其他模块 import 调用”两种场景下的导入问题
# - 直接运行：`python ./Potrace/color_trace.py ...` 时，`Potrace/` 会在 sys.path[0]，可直接导入 svg_stack
# - 被调用：`from Potrace.color_trace import color_trace` 时，需要显式把 Potrace 目录加入 sys.path
# -------------------------------------------------------------------------
_POTRACE_DIR = os.path.dirname(os.path.abspath(__file__))  # Potrace 模块目录
if _POTRACE_DIR not in sys.path:  # 避免重复插入
    sys.path.insert(0, _POTRACE_DIR)  # 确保 `svg_stack` 可被找到

try:  # 优先按脚本同目录的顶层包方式导入
    from svg_stack import svg_stack  # 导入 svg_stack 子模块
except ImportError:  # 若作为包导入失败，则使用相对导入兜底
    from .svg_stack import svg_stack  # type: ignore


# =============================================================================
# 全局变量 - 工具路径配置
# =============================================================================

# 工具路径：始终从“当前工作目录”出发（满足项目整体移动到任意目录仍可运行）
# 约定：你在项目根目录下运行脚本，因此 tools 位于 ./tools
_WORKDIR = os.path.abspath(os.getcwd())  # 当前工作目录（建议为项目根目录）
_TOOLS_DIR = os.path.join(_WORKDIR, 'tools')  # tools 根目录（相对工作目录）

# potrace 工具路径（工作目录下 ./tools/potrace/potrace.exe）
POTRACE_PATH = os.path.join(_TOOLS_DIR, 'potrace', 'potrace.exe')

# pngquant 工具路径（工作目录下 ./tools/pngquant/pngquant.exe）
PNGQUANT_PATH = os.path.join(_TOOLS_DIR, 'pngquant', 'pngquant.exe')

# ImageMagick 工具路径（工作目录下 ./tools/ImageMagick/magick.exe）
IMAGEMAGICK_PATH = os.path.join(_TOOLS_DIR, 'ImageMagick', 'magick.exe')

# POTRACE_PATH = os.path.join('./tools/potrace', 'potrace.exe')
# PNGQUANT_PATH = os.path.join('./tools/pngquant', 'pngquant.exe')
# IMAGEMAGICK_PATH = os.path.join('./tools/ImageMagick', 'magick.exe')

# pngnq 工具路径（可选，暂未使用本地路径）
PNGNQ_PATH = 'pngnq'

# 命令行最大长度限制
MAX_COMMAND_LENGTH = 1900

# 汇报级别（受 -v/--verbose 选项影响）
VERBOSE_LEVEL = 0

# 版本号
VERSION = '1.01'


# =============================================================================
# 模块一：命令执行模块
# =============================================================================

def verbose_print(*args, level=1):
    """
    根据汇报级别打印信息
    
    Args:
        *args: 要打印的内容
        level: 当前信息的汇报级别，默认为1
        
    Returns:
        None
    """
    global VERBOSE_LEVEL
    if VERBOSE_LEVEL >= level:
        print(*args)


def execute_command(command, stdinput=None, stdout_flag=False, stderr_flag=False):
    """
    在后台 shell 中运行命令，返回 stdout 和/或 stderr
    
    Args:
        command: 要运行的命令字符串
        stdinput: 发送到命令 stdin 的数据（字节），默认为 None
        stdout_flag: 是否接收命令的 stdout，默认为 False
        stderr_flag: 是否接收命令的 stderr，默认为 False
        
    Returns:
        stdout, stderr 或 (stdout, stderr) 元组，取决于参数设置
        如果两者都为 False，则返回 None
        
    Raises:
        Exception: 当命令返回非零退出码时抛出异常
    """
    # 设置管道
    stdin_pipe = subprocess.PIPE if stdinput is not None else None
    stdout_pipe = subprocess.PIPE if stdout_flag else None
    stderr_pipe = subprocess.PIPE
    
    verbose_print(f'命令：{command}')
    
    # 设置环境变量，确保使用正确的工具路径（必须覆盖可能残留的 color-trace 配置）
    env = os.environ.copy()  # 复制当前进程环境变量
    magick_dir = os.path.dirname(IMAGEMAGICK_PATH)  # magick.exe 所在目录（应为 ./tools/ImageMagick）

    # 强制覆盖 ImageMagick 相关环境变量，避免继续指向 color-trace 下的 ImageMagick
    env['MAGICK_HOME'] = magick_dir  # ImageMagick 根目录
    env['MAGICK_CONFIGURE_PATH'] = magick_dir  # 配置目录
    env['MAGICK_MODULE_PATH'] = os.path.join(magick_dir, 'modules')  # 模块目录
    env['MAGICK_CODER_MODULE_PATH'] = os.path.join(magick_dir, 'modules', 'coders')  # coder 模块目录
    env['MAGICK_CODER_FILTER_PATH'] = os.path.join(magick_dir, 'modules', 'filters')  # filter 模块目录

    # 优先使用我们的工具路径（把 tools 下目录放在 PATH 最前）
    env['PATH'] = (
        magick_dir + ';' +
        os.path.dirname(POTRACE_PATH) + ';' +
        os.path.dirname(PNGQUANT_PATH) + ';' +
        env.get('PATH', '')
    )
    
    # 创建并执行进程
    process = subprocess.Popen(
        shlex.split(command),
        stdin=stdin_pipe,
        stderr=stderr_pipe,
        stdout=stdout_pipe,
        shell=True,
        env=env
    )
    
    # 等待命令完成
    stdoutput, stderror = process.communicate(input=stdinput)
    return_code = process.wait()
    
    # 检查返回码
    if return_code != 0:
        # 尝试多种编码方式解码错误信息
        error_msg = None
        for encoding in [sys.getfilesystemencoding(), 'utf-8', 'gbk', 'cp936', 'latin-1']:
            try:
                error_msg = stderror.decode(encoding=encoding)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        if error_msg is None:
            error_msg = str(stderror)
        raise Exception(error_msg)
    
    # 根据参数返回结果
    if stdout_flag and not stderr_flag:
        return stdoutput
    elif stderr_flag and not stdout_flag:
        return stderror
    elif stdout_flag and stderr_flag:
        return (stdoutput, stderror)
    else:
        return None


# =============================================================================
# 模块二：图像处理模块
# =============================================================================

def rescale_image(source, destination, scale, filter_type='lanczos'):
    """
    使用 ImageMagick 将图片重新缩放、转为 png 格式
    
    Args:
        source: 源图像路径
        destination: 目标图像保存路径
        scale: 缩放比例，1.0 表示不缩放
        filter_type: 缩放滤镜类型，默认为 'lanczos'
        
    Returns:
        None
    """
    if scale == 1.0:
        # 不缩放，检查格式
        if os.path.splitext(source)[1].lower() not in ['.png']:
            # 非 png 格式则转换格式
            command = f'"{IMAGEMAGICK_PATH}" convert "{source}" "{destination}"'
            execute_command(command)
        else:
            # png 格式则直接复制
            shutil.copyfile(source, destination)
    else:
        # 执行缩放
        command = '"{magick}" convert "{src}" -filter {filter} -resize {resize}% "{dest}"'.format(
            magick=IMAGEMAGICK_PATH,
            src=source,
            filter=filter_type,
            resize=scale * 100,
            dest=destination
        )
        execute_command(command)


def quantize_image(source, quant_dest, color_count, algorithm='mc', dither=None):
    """
    将源图像量化到指定数量的颜色，保存到目标路径
    
    Args:
        source: 源图像路径，必须是 png 文件
        quant_dest: 量化后输出图像的路径
        color_count: 要缩减到的颜色数量，0 表示不量化
        algorithm: 量化算法，可选值：
            - 'mc': median-cut 中切（默认，使用 pngquant）
            - 'as': adaptive spatial subdivision 自适应空间细分（使用 ImageMagick）
            - 'nq': neuquant 神经量化（使用 pngnq）
        dither: 量化时使用的抖动拟色算法，可选值：
            - None: 默认，不拟色
            - 'floydsteinberg': 适用于 'mc', 'as', 'nq'
            - 'riemersma': 仅适用于 'as'
            
    Returns:
        None
        
    Raises:
        ValueError: 当使用错误的拟色类型时抛出
        NotImplementedError: 当使用未知的量化算法时抛出
    """
    if color_count in [0, 1]:
        # 跳过量化，直接复制
        shutil.copyfile(source, quant_dest)
        return
    
    if algorithm == 'mc':  # median-cut 中切
        if dither is None:
            dither_option = '--nofs'
        elif dither == 'floydsteinberg':
            dither_option = ''
        else:
            raise ValueError(f"对 'mc' 量化方法使用了错误的拟色类型：'{dither}'")
        
        # pngquant 不能保存到中文路径，使用 stdin/stdout 操作
        command = f'"{PNGQUANT_PATH}" --force {dither_option} {color_count} - < "{source}" > "{quant_dest}"'
        execute_command(command)
        
    elif algorithm == 'as':  # adaptive spatial subdivision
        if dither is None:
            dither_option = 'None'
        elif dither in ('floydsteinberg', 'riemersma'):
            dither_option = dither
        else:
            raise ValueError(f"对 'as' 量化方法使用了错误的拟色类型：'{dither}'")
        
        command = '"{magick}" convert "{src}" -dither {dither} -colors {colors} "{dest}"'.format(
            magick=IMAGEMAGICK_PATH,
            src=source,
            dither=dither_option,
            colors=color_count,
            dest=quant_dest
        )
        execute_command(command)
        
    elif algorithm == 'nq':  # neuquant
        ext = "~quant.png"
        dest_dir = os.path.dirname(quant_dest)
        
        if dither is None:
            dither_option = ''
        elif dither == 'floydsteinberg':
            dither_option = '-Q f '
        else:
            raise ValueError(f"对 'nq' 量化方法使用了错误的拟色类型：'{dither}'")
        
        command = '"{pngnq}" -f {dither}-d "{destdir}" -n {colors} -e {ext} "{src}"'.format(
            pngnq=PNGNQ_PATH,
            dither=dither_option,
            destdir=dest_dir,
            colors=color_count,
            ext=ext,
            src=source
        )
        execute_command(command)
        
        # pngnq 不支持保存到自定义目录，需要移动文件
        old_output = os.path.join(dest_dir, os.path.splitext(os.path.basename(source))[0] + ext)
        os.rename(old_output, quant_dest)
        
    else:
        raise NotImplementedError(f'未知的量化算法 "{algorithm}"')


def remap_image_with_palette(source, remap_dest, palette_image, dither=None):
    """
    用调色板图像的颜色重映射源图像
    
    Args:
        source: 源图像路径
        remap_dest: 输出保存路径
        palette_image: 调色板图像路径，包含源图像将重映射的颜色
        dither: 重映射时的拟色算法，可选值：
            - None: 默认，不拟色
            - 'floydsteinberg': Floyd-Steinberg 抖动
            - 'riemersma': Riemersma 抖动
            
    Returns:
        None
    
    Raises:
        IOError: 当调色板图像不存在时抛出
        ValueError: 当使用错误的拟色类型时抛出
    """
    # 检查调色板图像是否存在
    if not os.path.exists(palette_image):
        raise IOError(f"未找到重映射调色板：{palette_image}")
    
    # 设置拟色选项
    if dither is None:
        dither_option = 'None'
    elif dither in ('floydsteinberg', 'riemersma'):
        dither_option = dither
    else:
        raise ValueError(f"不合理的重映射拟色类型：'{dither}'")
    
    # 执行重映射命令
    command = '"{magick}" convert "{src}" -dither {dither} -remap "{palette}" "{dest}"'.format(
        magick=IMAGEMAGICK_PATH,
        src=source,
        dither=dither_option,
        palette=palette_image,
        dest=remap_dest
    )
    execute_command(command)


# =============================================================================
# 模块三：颜色处理模块
# =============================================================================

def create_color_table(source_image):
    """
    从源图像提取特征色，返回十六进制颜色列表
    
    Args:
        source_image: 源图像路径
        
    Returns:
        list: 十六进制颜色字符串列表，格式为 ['#rrggbb', ...]
    """
    # 使用 ImageMagick 提取唯一颜色
    command = f'"{IMAGEMAGICK_PATH}" "{source_image}" -unique-colors txt:-'
    stdoutput = execute_command(command, stdout_flag=True)
    
    # 解析输出中的颜色值
    pattern = '#[0-9A-F]{6}'
    im_output = stdoutput.decode(sys.getfilesystemencoding())
    hex_colors = re.findall(pattern, im_output)
    
    return hex_colors


def get_non_palette_color(palette, start_from_black=True, avoid_colors=None):
    """
    返回一个不在调色板内的十六进制颜色字符串
    
    Args:
        palette: 调色板颜色列表
        start_from_black: 是否从黑色开始搜索，False 则从白色开始
        avoid_colors: 需要规避的颜色列表
        
    Returns:
        str: 不在调色板内的十六进制颜色字符串
        
    Raises:
        Exception: 当调色板包含所有颜色时抛出异常
    """
    # 合并调色板和规避颜色
    if avoid_colors is None:
        final_palette = tuple(palette)
    else:
        final_palette = tuple(palette) + tuple(avoid_colors)
    
    # 确定搜索范围
    if start_from_black:
        color_range = range(int('ffffff', 16))
    else:
        color_range = range(int('ffffff', 16), 0, -1)
    
    # 查找不在调色板中的颜色
    for i in color_range:
        color = "#{0:06x}".format(i)
        if color not in final_palette:
            return color
    
    raise Exception("未能找到调色板之外的颜色")


def isolate_color(source, temp_file, dest_layer, target_color, palette, stack=False):
    """
    将指定颜色区域替换为黑色，其他区域为白色
    
    Args:
        source: 源图像路径，必须匹配调色板的颜色
        temp_file: 临时文件路径
        dest_layer: 输出图像的路径
        target_color: 要孤立的颜色（来自调色板）
        palette: 调色板列表，格式为 ["#rrggbb", ...]
        stack: 如果 True，颜色索引之前的颜色为白，之后的为黑
        
    Returns:
        None
    """
    color_index = palette.index(target_color)
    
    # 选择不在调色板中的背景色和前景色
    bg_white = "#FFFFFF"
    fg_black = "#000000"
    bg_near_white = get_non_palette_color(palette, False, (bg_white, fg_black))
    fg_near_black = get_non_palette_color(palette, True, (bg_near_white, bg_white, fg_black))
    
    # 打开源文件
    with open(source, 'rb') as src_file:
        stdinput = src_file.read()
    
    # 构建长命令以提高效率
    last_iteration = len(palette) - 1
    command_prefix = '"{magick}" convert "{src}" '.format(magick=IMAGEMAGICK_PATH, src=source)
    command_suffix = ' "{target}"'.format(target=temp_file)
    command_middle = ''
    
    for i, color in enumerate(palette):
        # 确定填充颜色
        if i == color_index:
            fill_color = fg_near_black
        elif i > color_index and stack:
            fill_color = fg_near_black
        else:
            fill_color = bg_near_white
        
        command_middle += ' -fill "{fill}" -opaque "{color}"'.format(fill=fill_color, color=color)
        
        # 当命令达到最大长度或最后一次迭代时执行
        if len(command_middle) >= MAX_COMMAND_LENGTH or (i == last_iteration and command_middle):
            command = command_prefix + command_middle + command_suffix
            stdoutput = execute_command(command, stdinput=stdinput, stdout_flag=True)
            stdinput = stdoutput
            command_middle = ''
    
    # 将前景变黑，背景变白
    command = '"{magick}" convert "{src}" -fill "{fillbg}" -opaque "{colorbg}" -fill "{fillfg}" -opaque "{colorfg}" "{dest}"'.format(
        magick=IMAGEMAGICK_PATH,
        src=temp_file,
        fillbg=bg_white,
        colorbg=bg_near_white,
        fillfg=fg_black,
        colorfg=fg_near_black,
        dest=dest_layer
    )
    execute_command(command, stdinput=stdinput)


def fill_with_color(source, destination):
    """
    用黑色填充图像的非透明区域
    
    Args:
        source: 源图像路径
        destination: 目标图像路径
        
    Returns:
        None
    """
    command = '"{magick}" convert "{src}" -fill "{color}" +opaque none "{dest}"'.format(
        magick=IMAGEMAGICK_PATH,
        src=source,
        color="#000000",
        dest=destination
    )
    execute_command(command)


# =============================================================================
# 模块四：描摹模块
# =============================================================================

def get_image_width(source):
    """
    获取图像宽度（像素）
    
    Args:
        source: 源图像路径
        
    Returns:
        int: 图像宽度（像素）
    """
    command = '"{magick}" identify -ping -format "%w" "{src}"'.format(
        magick=IMAGEMAGICK_PATH,
        src=source
    )
    stdoutput = execute_command(command, stdout_flag=True)
    width = int(stdoutput)
    return width


def trace_image(source, trace_dest, output_color, despeckle=2, smooth_corners=1.0, 
                optimize_paths=0.2, width=None, height=None, resolution=None):
    """
    使用 potrace 在指定的颜色和选项下进行矢量描摹
    
    Args:
        source: 源文件路径
        trace_dest: 输出目标文件路径
        output_color: 描摹路径填充的颜色（十六进制）
        despeckle: 抑制指定像素数量的斑点（等同于 potrace --turdsize）
        smooth_corners: 平滑转角参数，0 表示不平滑，1.334 为最大（等同于 potrace --alphamax）
        optimize_paths: 贝塞尔曲线优化参数，0 最小，5 最大（等同于 potrace --opttolerance）
        width: 输出的 svg 像素宽度，默认 None 保持原始比例
        height: 输出的 svg 像素高度
        resolution: 输出分辨率
        
    Returns:
        None
    """
    # 构建可选参数
    width_param = f'--width {width}' if width is not None else ''
    height_param = f'--height {height}' if height is not None else ''
    resolution_param = f'--resolution {resolution}' if resolution is not None else ''
    
    # 构建 potrace 命令
    command = '''"{potrace}" --svg -o "{dest}" -C "{color}" -t {despeckle} -a {smooth} -O {optimize} 
                {width} {height} {resolution} "{src}"'''.format(
        potrace=POTRACE_PATH,
        dest=trace_dest,
        color=output_color,
        despeckle=despeckle,
        smooth=smooth_corners,
        optimize=optimize_paths,
        width=width_param,
        height=height_param,
        resolution=resolution_param,
        src=source
    )
    
    verbose_print(command)
    execute_command(command)


# =============================================================================
# 模块五：文件处理模块
# =============================================================================

def check_range(min_val, max_val, type_func, type_name, str_val):
    """
    对 argparse 的参数，检查参数是否符合范围
    
    Args:
        min_val: 可接受的最小值
        max_val: 可接受的最大值
        type_func: 值转换函数，如 float, int
        type_name: 值的类型名称，如 "an integer"
        str_val: 包含期待值的字符串
        
    Returns:
        转换后的值
        
    Raises:
        argparse.ArgumentTypeError: 当值不符合要求时抛出
    """
    try:
        val = type_func(str_val)
    except ValueError:
        msg = "must be {typename}".format(typename=type_name)
        raise argparse.ArgumentTypeError(msg)
    
    if (max_val is not None) and (not min_val <= val <= max_val):
        msg = "must be between {min} and {max}".format(min=min_val, max=max_val)
        raise argparse.ArgumentTypeError(msg)
    elif not min_val <= val:
        msg = "must be {min} or greater".format(min=min_val)
        raise argparse.ArgumentTypeError(msg)
    
    return val


def escape_brackets(string):
    """
    转义字符串中的方括号，用于 glob 模式匹配
    
    Args:
        string: 原始字符串
        
    Returns:
        str: 转义后的字符串
    """
    letters = list(string)
    for i, letter in enumerate(letters[:]):
        if letter == '[':
            letters[i] = '[[]'
        elif letter == ']':
            letters[i] = '[]]'
    return ''.join(letters)


def get_input_output_pairs(arg_inputs, output_pattern="{0}.svg", ignore_duplicates=True):
    """
    使用 shell 通配符展开，得到 (input, output) 的遍历器
    
    Args:
        arg_inputs: 命令行输入参数，可包含 *? 通配符
        output_pattern: 输出文件命名模式，{0} 表示输入文件的基本名称
        ignore_duplicates: 是否忽略重复的输入文件
        
    Yields:
        tuple: (input_path, output_path) 元组
    """
    old_inputs = set()
    
    for arg_input in arg_inputs:
        if '*' in arg_input or '?' in arg_input:
            # 处理方括号转义
            if '[' in arg_input or ']' in arg_input:
                arg_input = escape_brackets(arg_input)
            inputs_ = tuple(iglob(os.path.abspath(arg_input)))
        else:
            # 确保非现有文件路径被包含，以便报告错误
            inputs_ = (arg_input,)
        
        for input_ in inputs_:
            if ignore_duplicates:
                if input_ not in old_inputs:
                    old_inputs.add(input_)
                    basename = os.path.basename(os.path.splitext(input_)[0])
                    output = output_pattern.format(basename)
                    yield input_, output
            else:
                basename = os.path.basename(os.path.splitext(input_)[0])
                output = output_pattern.format(basename)
                yield input_, output


def delete_files(*filepaths):
    """
    删除指定的文件（如果存在）
    
    Args:
        *filepaths: 要删除的文件路径列表
        
    Returns:
        None
    """
    for f in filepaths:
        if os.path.exists(f):
            os.remove(f)


# =============================================================================
# 模块六：多进程处理模块
# =============================================================================

def queue1_task(queue2, total_count, layers, settings, file_index, input_file, output):
    """
    队列1任务：初始化文件、重新缩放、缩减颜色
    
    Args:
        queue2: 第二个任务队列（颜色孤立 + 描摹）
        total_count: 用于测量队列二任务总数的值
        layers: 已排序的 svg 图层文件列表
        settings: 设置字典，包含以下键：
            - colors: 颜色数
            - quantization: 量化算法
            - dither: 拟色算法
            - remap: 重映射调色板
            - prescale: 预缩放比例
            - tmp: 临时文件目录
            - width: 输出宽度
            - height: 输出高度
            - resolution: 分辨率
        file_index: 输入文件的整数索引
        input_file: 输入 png 文件路径
        output: 输出 svg 路径
        
    Returns:
        None
    """
    # 创建输出目录
    dest_folder = os.path.dirname(os.path.abspath(output))
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # 设置临时文件路径
    scaled_file = os.path.abspath(os.path.join(settings['tmp'], '{0}~scaled.png'.format(file_index)))
    reduced_file = os.path.abspath(os.path.join(settings['tmp'], '{0}~reduced.png'.format(file_index)))
    
    try:
        # 选择缩放滤镜
        if settings['colors'] == 0:
            filter_type = 'point'
        else:
            filter_type = 'lanczos'
        
        # 执行缩放
        rescale_image(input_file, scaled_file, settings['prescale'], filter_type=filter_type)
        
        # 执行颜色量化或重映射
        if settings['colors'] is not None:
            quantize_image(scaled_file, reduced_file, settings['colors'], 
                          algorithm=settings['quantization'], dither=settings['dither'])
        elif settings['remap'] is not None:
            remap_image_with_palette(scaled_file, reduced_file, settings['remap'], 
                                     dither=settings['dither'])
        else:
            raise Exception("至少应该设置 'colors' 、 'remap' 中最少一个参数")
        
        # 创建颜色表
        if settings['colors'] == 1:
            color_table = ['#000000']
        else:
            color_table = create_color_table(reduced_file)
        
        # 更新任务总数
        if settings['colors'] is not None:
            total_count.value -= settings['colors'] - len(color_table)
        else:
            total_count.value -= settings['palette_color_count'] - len(color_table)
        
        # 初始化图层
        layers[file_index] += [False] * len(color_table)
        
        # 获取图像宽度
        width = settings['width'] if settings['width'] else f'{get_image_width(input_file)}pt'
        height = settings['height']
        resolution = settings['resolution']
        
        # 添加任务到第二个队列
        for i, color in enumerate(color_table):
            queue2.put({
                'width': width,
                'height': height,
                'resolution': resolution,
                'color': color,
                'palette': color_table,
                'reduced_image': reduced_file,
                'output_path': output,
                'file_index': file_index,
                'color_index': i
            })
    
    except (Exception, KeyboardInterrupt) as e:
        raise e
    else:
        # 删除临时文件
        delete_files(scaled_file)


def queue2_task(layers, layers_lock, settings, width, height, resolution, color, 
                palette, file_index, color_index, reduced_image, output_path):
    """
    队列2任务：分离颜色并描摹
    
    Args:
        layers: 有序列表，包含 svg 文件的描摹图层
        layers_lock: 读取和写入图层对象时的锁
        settings: 设置字典，包含以下键：
            - stack: 是否堆栈描摹
            - despeckle: 抑制斑点大小
            - smoothcorners: 平滑转角参数
            - optimizepaths: 路径优化参数
            - tmp: 临时文件目录
            - background: 是否设置背景
        width: 输入图像的宽度
        height: 输入图像的高度
        resolution: 分辨率
        color: 要孤立的颜色
        palette: 调色板列表
        file_index: 输入文件的整数索引
        color_index: 颜色的整数索引
        reduced_image: 已缩减颜色的输入图像
        output_path: 输出路径，svg 文件
        
    Returns:
        None
    """
    # 设置临时文件路径
    isolated_file = os.path.abspath(os.path.join(
        settings['tmp'], '{0}-{1}~isolated.png'.format(file_index, color_index)))
    layer_file = os.path.abspath(os.path.join(
        settings['tmp'], '{0}-{1}~layer.ppm'.format(file_index, color_index)))
    trace_format = '{0}-{1}~trace.svg'
    trace_file = os.path.abspath(os.path.join(
        settings['tmp'], trace_format.format(file_index, color_index)))
    
    try:
        # 孤立颜色或填充背景
        if color_index == 0 and settings['background']:
            verbose_print("Index {}".format(color))
            fill_with_color(reduced_image, layer_file)
        else:
            isolate_color(reduced_image, isolated_file, layer_file, color, 
                         palette, stack=settings['stack'])
        
        # 执行描摹
        trace_image(layer_file, trace_file, color, settings['despeckle'],
                   settings['smoothcorners'], settings['optimizepaths'],
                   width, height, resolution)
    
    except (Exception, KeyboardInterrupt) as e:
        delete_files(reduced_image, isolated_file, layer_file, trace_file)
        raise e
    else:
        delete_files(isolated_file, layer_file)
    
    # 更新图层状态
    layers_lock.acquire()
    try:
        layers[file_index][color_index] = True
        is_last_one = False not in layers[file_index]
    finally:
        layers_lock.release()
    
    # 如果所有图层都已完成，保存 svg 文档
    if is_last_one:
        layout = svg_stack.CBoxLayout()
        
        trace_layers = [os.path.abspath(os.path.join(
            settings['tmp'], trace_format.format(file_index, l))) 
            for l in range(len(layers[file_index]))]
        
        for t in trace_layers:
            layout.addSVG(t)
        
        document = svg_stack.Document()
        document.setLayout(layout)
        
        with open(output_path, 'w') as f:
            document.save(f)
        
        delete_files(reduced_image, *trace_layers)


def process_worker(queue1, queue2, completed_count, total_count, layers, layers_lock, settings):
    """
    处理进程的工作函数
    
    Args:
        queue1: 第一个任务队列（缩放 + 颜色缩减）
        queue2: 第二个任务队列（颜色隔离 + 描摹）
        completed_count: 第二个队列已完成任务数
        total_count: 第二个队列总任务数
        layers: 嵌套列表，layers[file_index][color_index] 表示图层是否已描摹
        layers_lock: 读取和写入图层对象时的锁
        settings: 设置字典
        
    Returns:
        None
    """
    while True:
        # 优先处理第二个队列
        while not queue2.empty():
            try:
                task_params = queue2.get(block=False)
                queue2_task(layers, layers_lock, settings, **task_params)
                queue2.task_done()
                completed_count.value += 1
            except queue.Empty:
                break
        
        # 处理第一个队列
        try:
            task_params = queue1.get(block=False)
            queue1_task(queue2, total_count, layers, settings, **task_params)
            queue1.task_done()
        except queue.Empty:
            time.sleep(.01)
        
        # 检查是否完成
        if queue2.empty() and queue1.empty():
            break


# =============================================================================
# 模块七：核心功能模块
# =============================================================================

def color_trace(input_list, output_list, color_count, process_count, quantization='mc',
                dither=None, remap=None, stack=False, prescale=2, despeckle=2,
                smoothcorners=1.0, optimizepaths=0.2, background=False,
                width=None, height=None, resolution=None):
    """
    用指定选项彩色描摹输入图片
    
    Args:
        input_list: 输入文件列表，源 png 文件
        output_list: 输出文件列表，目标 svg 文件
        color_count: 要量化缩减到的颜色数量，0 表示不量化
        process_count: 图像处理进程数
        quantization: 量化算法：
            - 'mc': median-cut 中切（默认，使用 pngquant）
            - 'as': adaptive spatial subdivision（使用 ImageMagick）
            - 'nq': neuquant（使用 pngnq）
        dither: 量化时使用的抖动拟色算法：
            - None: 默认，不拟色
            - 'floydsteinberg': 适用于 'mc', 'as', 'nq'
            - 'riemersma': 仅适用于 'as'
        remap: 用于颜色缩减的自定义调色板图像路径（覆盖 color_count 和 quantization）
        stack: 是否堆栈彩色描摹（可以得到更精确的输出）
        prescale: 预缩放比例
        despeckle: 抑制指定像素数量的斑点
        smoothcorners: 平滑转角参数（0-1.334）
        optimizepaths: 贝塞尔曲线优化参数（0-5）
        background: 设置第一个颜色为整个 svg 背景
        width: 输出 svg 宽度
        height: 输出 svg 高度
        resolution: 输出分辨率
        
    Returns:
        None
    """
    # 创建临时目录
    tmp_dir = tempfile.mkdtemp()
    
    # 创建任务队列
    queue1 = multiprocessing.JoinableQueue()  # 缩放和颜色缩减
    queue2 = multiprocessing.JoinableQueue()  # 颜色分离和描摹
    
    # 创建共享管理器
    manager = multiprocessing.Manager()
    layers = []
    for i in range(min(len(input_list), len(output_list))):
        layers.append(manager.list())
    
    # 创建图层锁
    layers_lock = multiprocessing.Lock()
    
    # 创建共享计数器
    completed_count = multiprocessing.Value('i', 0)
    
    if color_count is not None:
        total_count = multiprocessing.Value('i', len(layers) * color_count)
    elif remap is not None:
        palette_color_count = len(create_color_table(remap))
        total_count = multiprocessing.Value('i', len(layers) * palette_color_count)
    else:
        raise Exception("应当提供 'colors' 和 'remap' 至少一个参数")
    
    # 构建设置字典
    settings = {
        'colors': color_count,
        'quantization': quantization,
        'dither': dither,
        'remap': remap,
        'stack': stack,
        'prescale': prescale,
        'despeckle': despeckle,
        'smoothcorners': smoothcorners,
        'optimizepaths': optimizepaths,
        'background': background,
        'width': width,
        'height': height,
        'resolution': resolution,
        'tmp': tmp_dir,
        'palette_color_count': palette_color_count if remap else None
    }
    
    # 创建并启动进程
    process_list = []
    for i in range(process_count):
        process = multiprocessing.Process(
            target=process_worker,
            args=(queue1, queue2, completed_count, total_count, layers, layers_lock, settings)
        )
        process.name = "color_trace worker #" + str(i)
        process.start()
        process_list.append(process)
    
    try:
        # 添加任务到第一个队列
        for index, (input_file, output) in enumerate(zip(input_list, output_list)):
            verbose_print(input_file, ' -> ', output)
            queue1.put({'input_file': input_file, 'output': output, 'file_index': index})
        
        # 显示进度
        while completed_count.value < total_count.value:
            sys.stdout.write("\r%.1f%%" % (completed_count.value / total_count.value * 100))
            sys.stdout.flush()
            time.sleep(0.25)
        
        sys.stdout.write("\rTracing complete!\n")
        
        # 等待队列完成
        queue1.join()
        queue2.join()
    
    except (Exception, KeyboardInterrupt) as e:
        for process in process_list:
            process.terminate()
        shutil.rmtree(tmp_dir)
        raise e
    
    # 清理
    for process in process_list:
        process.terminate()
    shutil.rmtree(tmp_dir)


# =============================================================================
# 模块八：命令行接口模块
# =============================================================================

def parse_arguments(cmdargs=None):
    """
    解析命令行参数
    
    Args:
        cmdargs: 如果指定，则使用这些参数，否则使用命令行参数
        
    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description="使用 potrace 将位图转化为彩色 svg 矢量图",
        add_help=False,
        prefix_chars='-/'
    )
    
    # 帮助选项
    parser.add_argument('-h', '--help', '/?',
                       action='help',
                       help="显示帮助")
    
    # 文件输入输出参数
    parser.add_argument('-i', '--input', metavar='src', nargs='+', required=True,
                       help="输入文件，支持 * 和 ? 通配符")
    parser.add_argument('-o', '--output', metavar='dest',
                       help="输出保存路径，支持 * 通配符")
    parser.add_argument('-d', '--directory', metavar='destdir',
                       help="输出保存的文件夹")
    
    # 处理参数
    parser.add_argument('-C', '--cores', metavar='N',
                       type=functools.partial(check_range, 0, None, int, "an integer"),
                       help="多进程处理的进程数（默认使用全部核心）")
    
    # 尺寸参数
    parser.add_argument('--width', metavar='<dim>',
                       help="输出 svg 图像宽度，例如：6.5in、15cm、100pt，默认单位是 inch")
    parser.add_argument('--height', metavar='<dim>',
                       help="输出 svg 图像高度，例如：6.5in、15cm、100pt，默认单位是 inch")
    
    # 颜色数和调色板互斥组
    color_palette_group = parser.add_mutually_exclusive_group(required=True)
    color_palette_group.add_argument('-c', '--colors', metavar='N',
                                    type=functools.partial(check_range, 0, 256, int, "an integer"),
                                    help="[若未使用 -p 参数，则必须指定该参数] "
                                         "表示在描摹前，先缩减到多少个颜色。最多 256 个。"
                                         "0 表示跳过缩减颜色（除非图片已经缩减过颜色，否则不推荐0）。")
    
    # 量化算法
    parser.add_argument('-q', '--quantization', metavar='algorithm',
                       choices=('mc', 'as', 'nq'), default='mc',
                       help="颜色量化算法：mc, as, or nq. "
                            "'mc' (Median-Cut，中切，默认); "
                            "'as' (Adaptive Spatial Subdivision，自适应空间细分); "
                            "'nq' (NeuQuant，神经量化)。")
    
    # 拟色算法互斥组
    dither_group = parser.add_mutually_exclusive_group()
    dither_group.add_argument('-fs', '--floydsteinberg', action='store_true',
                             help="启用 Floyd-Steinberg 拟色（适用于所有量化算法或 -p/--palette）。"
                                  "警告：任何拟色算法都会显著增加输出 svg 图片的大小和复杂度。")
    dither_group.add_argument('-ri', '--riemersma', action='store_true',
                             help="启用 Riemersma 拟色（只适用于 as 量化算法或 -p/--palette）")
    
    # 调色板参数
    color_palette_group.add_argument('-r', '--remap', metavar='paletteimg',
                                    help="使用一个自定义调色板图像，用于颜色缩减 [覆盖 -c 和 -q 选项]")
    
    # 图像选项
    parser.add_argument('-s', '--stack', action='store_true',
                       help="堆栈描摹（若要更精确的输出，推荐用这个）")
    parser.add_argument('-p', '--prescale', metavar='size',
                       type=functools.partial(check_range, 0, None, float, "a floating-point number"),
                       default=1,
                       help="为得到更多的细节，在描摹前，先将图片进行缩放（默认值：1）。"
                            "例如使用 2，描摹前先预放大两倍。")
    
    # potrace 选项
    parser.add_argument('-D', '--despeckle', metavar='size',
                       type=functools.partial(check_range, 0, None, int, "an integer"),
                       default=2,
                       help='抑制斑点的大小（单位是像素）（默认值：2）')
    parser.add_argument('-S', '--smoothcorners', metavar='threshold',
                       type=functools.partial(check_range, 0, 1.334, float, "a floating-point number"),
                       default=1.0,
                       help="转角平滑参数：0 表示不作平滑处理，1.334 是最大。（默认值：1.0）")
    parser.add_argument('-O', '--optimizepaths', metavar='tolerance',
                       type=functools.partial(check_range, 0, 5, float, "a floating-point number"),
                       default=0.2,
                       help="贝塞尔曲线优化参数：最小是 0，最大是 5（默认值：0.2）")
    parser.add_argument('-bg', '--background', action='store_true',
                       help="将第一个颜色设为背景色，并尽可能优化最终的 svg")
    
    # 其他选项
    parser.add_argument('-v', '--verbose', action='store_true',
                       help="打印出运行时的细节")
    parser.add_argument('--version', action='version',
                       version='%(prog)s {ver}'.format(ver=VERSION),
                       help='显示程序版本')
    
    # 解析参数
    if cmdargs is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmdargs)
    
    # 验证参数
    # 多输入时，output 必须包含 * 通配符
    multi_inputs = False
    for i, input_ in enumerate(get_input_output_pairs(args.input)):
        if i:
            multi_inputs = True
            break
    if multi_inputs and args.output is not None and '*' not in args.output:
        parser.error("argument -o/--output: must contain '*' wildcard when using multiple input files")
    
    # riemersma 拟色只允许与 'as' 量化一起使用
    if args.riemersma:
        if args.quantization != 'as' and args.remap is None:
            parser.error("argument -ri/--riemersma: only allowed with 'as' quantization")
    
    return args


def main(args=None):
    """
    主函数：收集参数并运行描摹
    
    Args:
        args: 参数对象，如果为 None 则从命令行解析
        
    Returns:
        None
    """
    if args is None:
        args = parse_arguments()
    
    # 设置汇报级别
    if args.verbose:
        global VERBOSE_LEVEL
        VERBOSE_LEVEL = 1
    
    # 设置输出文件名形式
    if args.output is None:
        output_pattern = "{0}.svg"
    elif '*' in args.output:
        output_pattern = args.output.replace('*', "{0}")
    else:
        output_pattern = args.output
    
    # 添加输出文件夹路径
    if args.directory is not None:
        dest_folder = args.directory.strip('\"\'')
        output_pattern = os.path.join(dest_folder, output_pattern)
    
    # 设置进程数
    if args.cores is None:
        try:
            process_count = multiprocessing.cpu_count()
        except NotImplementedError:
            verbose_print("无法确定CPU核心数，因此假定为 1")
            process_count = 1
    else:
        process_count = args.cores
    
    # 收集输入输出列表
    io_pairs = zip(*get_input_output_pairs(args.input, output_pattern))
    try:
        input_list, output_list = io_pairs
    except ValueError:
        input_list, output_list = [], []
    
    # 设置拟色算法
    if args.floydsteinberg:
        dither = 'floydsteinberg'
    elif args.riemersma:
        dither = 'riemersma'
    else:
        dither = None
    
    color_count = args.colors
    
    # 构建参数字典
    trace_params = vars(args)
    for k in ('colors', 'directory', 'input', 'output', 'cores', 
              'floydsteinberg', 'riemersma', 'verbose'):
        trace_params.pop(k, None)
    
    # 执行彩色描摹
    color_trace(input_list, output_list, color_count, process_count, 
                dither=dither, **trace_params)


if __name__ == '__main__':
    main()
