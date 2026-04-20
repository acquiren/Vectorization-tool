import os
import cv2
import numpy as np
import subprocess


def clean_dir(input_dir):
    """
    清理目录下所有文件
    """
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"删除 {file_path} 时出错: {e}")

def jpg2png(input_path, output_path):
    """
    将 JPG 图像转换为 PNG 格式(基于cv2)
    
    Args:
        input_path: 输入图像路径
        output_path: 输出 PNG 图像路径
    Returns:
        转换后的图像路径
    """
    try:
            # 检查输入文件是否存在
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        print(f"开始 {input_path} 转换为 PNG 格式并保存为 {output_path}")

        # 读取图像
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        # 检查图像是否读取成功
        if img is None:
            raise RuntimeError(f"无法读取图像: {input_path}")
        
        # 保存为 PNG 格式
        cv2.imwrite(output_path, img)
        print(f"✅成功将 {input_path} 转换为 PNG 格式并保存为 {output_path}")
        print('='*50 + '\n\n')
        return output_path
    except Exception as e:
        raise RuntimeError(f"转换失败: {e}")

def convert_4ch_to_3ch(input_path, output_path, replace_white_bg=True):
    """
    将四通道(RGBA)图片转为三通道(RGB)

    Args:
        input_path: 输入图片路径
        output_path: 输出图片路径
        replace_white_bg: 是否将透明区域替换为白色（True）/黑色（False）
    """
    # 读取图片（保留Alpha通道）
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"无法读取图片：{input_path}")

    # 情况1：已经是3通道（RGB），直接保存
    if len(img.shape) == 3 and img.shape[2] == 3:
        cv2.imwrite(output_path, img)
        print(f"✅ {input_path} 已是3通道，直接保存")
        return output_path

    # 情况2：4通道（RGBA），处理透明通道
    elif len(img.shape) == 3 and img.shape[2] == 4:
        # 分离RGB和Alpha通道
        bgr = img[:, :, :3]
        alpha = img[:, :, 3] / 255.0  # 归一化Alpha到0-1
        
        # 创建背景（白色/黑色）
        bg_color = [255, 255, 255] if replace_white_bg else [0, 0, 0]
        bg = np.ones_like(bgr, dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)
        
        # 混合前景和背景（Alpha融合）
        bgr = (alpha[:, :, np.newaxis] * bgr + (1 - alpha[:, :, np.newaxis]) * bg).astype(np.uint8)
        
        # 保存3通道图片
        cv2.imwrite(output_path, bgr)
        print(f"✅ 已转换：{input_path} → {output_path}")
        return output_path

def png2bmp(input_path, output_path):
    """
    将 PNG 或 JPG 图像转换为 BMP 格式
    
    Args:
        input_path: 输入图像路径
        output_path: 输出 BMP 图像路径
    Returns:
        转换后的图像路径
    """

    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    # 检查输入文件格式
    if input_path.lower().endswith('.bmp'):
        print(f"✅ 输入文件已是 BMP 格式: {input_path}")
        print('='*50 + '\n\n')
        return input_path

    elif not input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise ValueError("输入文件必须是 PNG 或 JPG 格式")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 检查输出文件格式
    if not output_path.lower().endswith('.bmp'):
        raise ValueError("输出文件必须是 BMP 格式")
    
    print(f"开始将 {input_path} 转换为 BMP 格式并保存为 {output_path}")
    
    # 读取图像并转换为 BMP
    try:
        # 读取图像
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        # 检查图像是否读取成功
        if img is None:
            raise RuntimeError(f"无法读取图像: {input_path}")
        
        # 保存为 BMP 格式
        cv2.imwrite(output_path, img)
        print(f"✅成功将 {input_path} 转换为 BMP 格式并保存为 {output_path}")
        print('='*50 + '\n\n')
        return output_path
    except Exception as e:
        raise RuntimeError(f"转换失败: {e}")

class PotraceRunner:
    def __init__(self):
        self.init()

    def init(self,input_path = None):
        """
        初始化PotraceRunner
        
        Args:
            input_path: 输入文件路径
        """
        self.input_path = input_path

    def get_tool_path(self, tool_name):
        """
        获取工具的绝对路径
        
        Args:
            tool_name: 工具名称（'potrace' 或 'pngquant'）
        
        Returns:
            工具的绝对路径
        """

        # 定义 potrace 工具的相对路径
        if tool_name == 'potrace' or tool_name == 'mkbitmap':
            default_path = os.path.join('.', 'tools', f'{tool_name}', f'{tool_name}.exe')
        else:
            raise ValueError(f"未知工具名称: {tool_name}")
        
        # 检查工具是否存在
        if os.path.exists(default_path):
            return default_path
        else:
            raise FileNotFoundError(f"potrace工具 {tool_name} 未找到")
        
    def potrace_run(self, t=2, a=1, o=0.3, z='minority',r=72, format='svg', input_path=None, output_path=None):
        '''
        运行 potrace 工具
        
        Args:
            t: 阈值参数（默认2），控制二值化的阈值，值越小越敏感
            a: 面积参数（默认1），控制输出平滑度，值越大输出越平滑
            o: 曲线优化容差（默认0.3），值越大精度越低
            z: 颜色选择参数（默认'minority'）
            r: 分辨率参数（默认72），控制svg占用显示屏大小，值越大占用越小
            format: 输出格式（默认'svg'）
            input_image: 输入图像路径（默认None）
            output_image: 输出图像路径（默认None）
        '''

        # 获取 potrace 工具路径
        tool_path = self.get_tool_path('potrace')
        print(f"已获取potrace 工具路径: {tool_path}")

        # img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        # if max(img.shape) > 1024:
            # r = max(img.shape)/8 # 设置分辨率

        if os.path.exists(tool_path):
            # 获取 potrace 命令
            commend = [tool_path] + ['-b', format] + ['-t', str(t), '-a', str(a), '-O', str(o), '-z', z] + ['-o', output_path, input_path]
            # 执行 potrace 命令
            result = subprocess.run(commend, capture_output=True, text=True)
        else:
            raise FileNotFoundError(f"potrace工具 {tool_path} 未找到")

        if result.returncode == 0:
            print("✅ potrace命令执行成功")
            print('='*50 + '\n\n')
            # print(result.stdout)
            return output_path
        else:
            print(f"potrace命令执行失败，返回码: {result.returncode}")
            print(result.stderr)