#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本 - 图形矢量化工具测试

功能：
1. 加载并显示UI界面
2. 导入图片并显示预览
3. 显示SVG文件到输出预览

使用方法：
1. 在 IDE 中打开此脚本
2. 直接运行脚本启动图形界面
"""

import os
import sys
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtSvg import QSvgRenderer
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

from log import Logger
from Potrace.color_trace import color_trace

# 全局变量
UI_FILE_PATH = os.path.join('./ui', 'vectorizer_ui.ui')


class VectorizerApp:
    """
    图形矢量化应用类
    
    提供打开UI、导入图片、显示SVG等功能
    """
    
    def __init__(self):
        """
        初始化应用
        """
        self.app = QApplication(sys.argv)
        self.ui = None
        self.file_path = None
        self.output_svg_path = './output/final_svg.svg'
        self.output_svg_dir = './output'
        self.logger = Logger()
        
        # 缩放相关属性
        self.current_input_pixmap = None  # 当前输入图片的pixmap
        self.current_output_pixmap = None  # 当前输出SVG的pixmap
        self.current_input_zoom = 100  # 当前输入缩放百分比
        self.current_output_zoom = 100  # 当前输出缩放百分比
        
        # 确保输出目录存在
        os.makedirs(self.output_svg_dir, exist_ok=True)
        
    def open_ui(self):
        """
        打开并加载UI文件
        
        Returns:
            ui: 加载的UI对象
        """
        try:
            self.ui = loadUi(UI_FILE_PATH)
            
            # 连接缩放滑块信号
            self.ui.slider_zoom_input.valueChanged.connect(self.on_zoom_input_changed)
            self.ui.slider_zoom_output.valueChanged.connect(self.on_zoom_output_changed)
            
            # 连接按钮信号
            self.ui.btn_import.clicked.connect(self.import_image)
            self.ui.btn_start.clicked.connect(self.show_svg)
            
            self.logger.write_log("UI界面加载成功")
            return self.ui
        except Exception as e:
            self.logger.write_log(f"UI界面加载失败: {str(e)}")
            raise
    
    def import_image(self):
        """
        导入图片并显示到输入预览区域
        
        Returns:
            file_path: 导入的图片路径
        """
        try:
            # 打开文件选择对话框
            file_path, _ = QFileDialog.getOpenFileName(
                self.ui, "导入图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp)"
            )
            
            if not file_path:
                return None
            
            # 保存文件地址
            self.file_path = file_path
            self.output_svg_path = os.path.join(self.output_svg_dir, 'final_svg.svg')
            
            # 显示图片到预览区域
            pixmap = QPixmap(file_path)
            self.current_input_pixmap = pixmap  # 保存原始pixmap
            self.update_input_zoom()  # 应用当前缩放
            
            self.ui.lbl_preview_input.setText("")
            
            self.logger.write_log(f"导入图片: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.write_log(f"导入文件失败: {str(e)}")
            QMessageBox.warning(self.ui, "错误", f"导入文件失败: {str(e)}")
            return None
    
    def show_svg(self):
        """
        显示SVG文件到输出预览区域
        
        Returns:
            bool: 是否显示成功
        """
        try:
            # 检查是否有导入的图片
            if not self.file_path:
                self.ui.lbl_preview_output.setText("无法显示图片，请先导入")
                return False
            
            # 构建SVG文件路径
            svg_path = self.output_svg_path
            
            print(f"[调试] SVG文件路径: {svg_path}")
            
            # 检查SVG文件是否存在
            if not os.path.exists(svg_path):
                error_msg = f"SVG文件不存在\n路径: {svg_path}"
                print(f"[错误] {error_msg}")
                self.ui.lbl_preview_output.setText("SVG文件不存在")
                QMessageBox.warning(self.ui, "错误", error_msg)
                return False
            
            # 检查文件大小
            file_size = os.path.getsize(svg_path)
            print(f"[调试] SVG文件大小: {file_size} 字节")
            
            if file_size == 0:
                error_msg = "SVG文件为空"
                print(f"[错误] {error_msg}")
                self.ui.lbl_preview_output.setText(error_msg)
                QMessageBox.warning(self.ui, "错误", error_msg)
                return False
            
            # 创建SVG渲染器
            renderer = QSvgRenderer(svg_path)
            
            # 检查渲染器是否有效
            if not renderer.isValid():
                error_msg = f"SVG文件格式无效\n文件: {svg_path}"
                print(f"[错误] {error_msg}")
                self.ui.lbl_preview_output.setText("SVG文件格式无效")
                QMessageBox.warning(self.ui, "错误", error_msg)
                return False
            
            # 获取预览标签
            preview_label = self.ui.lbl_preview_output
            
            # 获取SVG默认尺寸
            svg_size = renderer.defaultSize()
            
            if svg_size.isEmpty():
                svg_size = preview_label.size()
            
            # 创建与SVG尺寸匹配的Pixmap
            pixmap = QPixmap(svg_size)
            
            # 检查Pixmap是否创建成功
            if pixmap.isNull():
                error_msg = "无法创建Pixmap"
                print(f"[错误] {error_msg}")
                self.ui.lbl_preview_output.setText(error_msg)
                QMessageBox.warning(self.ui, "错误", error_msg)
                return False
            
            pixmap.fill(Qt.white)  # 填充白色背景
            
            # 创建QPainter并在Pixmap上渲染SVG
            from PyQt5.QtGui import QPainter
            painter = QPainter(pixmap)
            renderer.render(painter)
            painter.end()
            
            # 缩放并显示到预览标签
            scaled_pixmap = pixmap.scaled(
                preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.current_output_pixmap = pixmap  # 保存原始SVG pixmap
            self.update_output_zoom()  # 应用当前缩放
            
            preview_label.setText("")
            
            self.logger.write_log(f"显示SVG: {svg_path}")
            return True
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"[异常] {error_detail}")
            
            error_msg = f"加载SVG失败:\n{str(e)}\n\n详细信息:\n{error_detail}"
            self.ui.lbl_preview_output.setText(f"加载SVG失败: {str(e)}")
            QMessageBox.critical(self.ui, "错误", error_msg)
            return False
    
    def run(self):
        """
        运行应用程序
        
        Returns:
            int: 应用程序退出码
        """
        self.open_ui()
        self.ui.btn_import.clicked.connect(self.import_image)
        self.ui.btn_start.clicked.connect(self.show_svg)
        self.ui.show()
        return self.app.exec_()
    
    def on_zoom_input_changed(self, value):
        """
        输入图片缩放滑块变化处理
        
        Args:
            value: 滑块值（0-200）
        """
        self.current_input_zoom = value
        self.ui.lbl_zoom_input_value.setText(f"{value}%")
        self.update_input_zoom()
    
    def on_zoom_output_changed(self, value):
        """
        输出SVG缩放滑块变化处理
        
        Args:
            value: 滑块值（0-200）
        """
        self.current_output_zoom = value
        self.ui.lbl_zoom_output_value.setText(f"{value}%")
        self.update_output_zoom()
    
    def update_input_zoom(self):
        """
        更新输入图片的缩放显示
        """
        if self.current_input_pixmap is None:
            return
        
        # 计算缩放比例
        scale_factor = self.current_input_zoom / 100.0
        
        # 应用缩放
        preview_label = self.ui.lbl_preview_input
        scaled_pixmap = self.current_input_pixmap.scaled(
            int(self.current_input_pixmap.width() * scale_factor),
            int(self.current_input_pixmap.height() * scale_factor),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        # 显示缩放后的图片
        preview_label.setPixmap(scaled_pixmap)
    
    def update_output_zoom(self):
        """
        更新输出SVG的缩放显示
        """
        if self.current_output_pixmap is None:
            return
        
        # 计算缩放比例
        scale_factor = self.current_output_zoom / 100.0
        
        # 应用缩放
        preview_label = self.ui.lbl_preview_output
        scaled_pixmap = self.current_output_pixmap.scaled(
            int(self.current_output_pixmap.width() * scale_factor),
            int(self.current_output_pixmap.height() * scale_factor),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        # 显示缩放后的图片
        preview_label.setPixmap(scaled_pixmap)


def main():
    """
    主入口函数
    """
    app = VectorizerApp()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
