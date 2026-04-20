#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图形矢量化工具 - PyQt5 启动脚本

使用方法：
1. 在 IDE 中打开此脚本
2. 直接运行脚本启动图形界面
"""

import os
import sys
import argparse
import subprocess
import multiprocessing
from skimage import io
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtSvg import QSvgRenderer
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

# =========================导入自定义库============================= #
from Potrace import potrace_vector
from DiffVG.refine_svg import refine_svg
from SuperSVG.inference import bitmap_to_svg
from Potrace.color_trace import color_trace


# =========================配置全局变量============================= #
# 界面文件路径
UI_FILE_PATH = os.path.join('./ui', 'vectorizer_ui.ui')
# 工具文件路径
POTRACE_PATH = os.path.join('./tools/potrace', 'potrace.exe')
PNGQUANT_PATH = os.path.join('./tools/pngquant', 'pngquant.exe')
IMAGEMAGICK_PATH = os.path.join('./tools/ImageMagick', 'magick.exe')

# print(UI_FILE_PATH)
# print(POTRACE_PATH)
# print(PNGQUANT_PATH)
# print(IMAGEMAGICK_PATH)
# ===============================================================

# 移动文件位置
def move_svg_os(source_path, target_dir):
    """
    使用os.rename移动SVG文件
    
    Args:
        source_path: 源SVG文件路径
        target_dir: 目标目录路径
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    file_name = 'final_svg.svg'
    target_path = os.path.join(target_dir, file_name)
    
    os.rename(source_path, target_path)
    print(f"SVG文件已移动到: {target_path}")

# 界面缩放比例
SCALE_FACTOR = 1.3

class App(QApplication):
    def __init__(self, sys_argv):
        # 初始化实例变量
        super().__init__(sys_argv)
        self.setStyle('Fusion')
        self.file_path = None
        self.tmp_dir = './tmp'
        self.output_svg_path = None
        self.output_svg_dir = './output'
        # self.parameter_init()

        # 清理残留文件
        potrace_vector.clean_dir(self.output_svg_dir)
        potrace_vector.clean_dir(self.tmp_dir)
        potrace_vector.clean_dir(f'{self.tmp_dir}/refine_svg')
        
        # 初始化界面
        self.interface_init()


# ====================================================== #
# 初始化
# ====================================================== #

    # 初始化界面
    def interface_init(self):
        # 加载界面文件
        self.ui = uic.loadUi(UI_FILE_PATH)


        # ====================================================
        # 基础模式
        # ====================================================

        # 实现参数联动
        # 色彩保留
        self.ui.slider_color.valueChanged.connect(self.ui.spinbox_color.setValue)
        self.ui.spinbox_color.valueChanged.connect(self.ui.slider_color.setValue)
        # 细节优化
        self.ui.slider_detail.valueChanged.connect(self.ui.spinbox_detail.setValue)
        self.ui.spinbox_detail.valueChanged.connect(self.ui.slider_detail.setValue)
        # SuperSVG 
        self.ui.slider_supersvg.valueChanged.connect(self.ui.spinbox_supersvg.setValue)
        self.ui.spinbox_supersvg.valueChanged.connect(self.ui.slider_supersvg.setValue)


        # ======================================================
        # 高级模式
        # ======================================================

        # Potrace 参数（整数滑块 ↔ 数值框）
        self.ui.slider_potrace_turd.valueChanged.connect(self.ui.spinbox_potrace_turd.setValue)
        self.ui.spinbox_potrace_turd.valueChanged.connect(self.ui.slider_potrace_turd.setValue)
        # _turn_keys = (
        #     "black", "white", "right", "left", "minority", "majority", "random",
        # )
        # for i, key in enumerate(_turn_keys):
        #     self.ui.comboBox_potrace_turnpolicy.setItemData(i, key, Qt.UserRole)

        def _potrace_smooth_from_slider(v):
            self.ui.doubleSpinBox_potrace_smooth.blockSignals(True)
            self.ui.doubleSpinBox_potrace_smooth.setValue(v * 0.1)
            self.ui.doubleSpinBox_potrace_smooth.blockSignals(False)
        def _potrace_smooth_from_spin(val):
            self.ui.slider_potrace_smooth.blockSignals(True)
            self.ui.slider_potrace_smooth.setValue(int(round(val * 10)))
            self.ui.slider_potrace_smooth.blockSignals(False)
        def _potrace_opt_from_slider(v):
            self.ui.doubleSpinBox_potrace_opttolerance.blockSignals(True)
            self.ui.doubleSpinBox_potrace_opttolerance.setValue(v * 0.1)
            self.ui.doubleSpinBox_potrace_opttolerance.blockSignals(False)
        def _potrace_opt_from_spin(val):
            self.ui.slider_potrace_opttolerance.blockSignals(True)
            self.ui.slider_potrace_opttolerance.setValue(int(round(val * 10)))
            self.ui.slider_potrace_opttolerance.blockSignals(False)

        self.ui.slider_potrace_smooth.valueChanged.connect(_potrace_smooth_from_slider)
        self.ui.doubleSpinBox_potrace_smooth.valueChanged.connect(_potrace_smooth_from_spin)
        self.ui.slider_potrace_opttolerance.valueChanged.connect(_potrace_opt_from_slider)
        self.ui.doubleSpinBox_potrace_opttolerance.valueChanged.connect(_potrace_opt_from_spin)

        # diffvg优化（与基础页「细节优化」相同联动）
        self.ui.slider_advanced_diffvg.valueChanged.connect(self.ui.spinbox_advanced_diffvg.setValue)
        self.ui.spinbox_advanced_diffvg.valueChanged.connect(self.ui.slider_advanced_diffvg.setValue)
        

        # 彩色矢量化参数(滑块 ↔ 数值框)
        self.ui.slider_color_count.valueChanged.connect(self.ui.spinBox_color_count.setValue)
        self.ui.spinBox_color_count.valueChanged.connect(self.ui.slider_color_count.setValue)
        self.ui.slider_color_despeckle.valueChanged.connect(self.ui.spinBox_color_despeckle.setValue)
        self.ui.spinBox_color_despeckle.valueChanged.connect(self.ui.slider_color_despeckle.setValue)

        def _color_trace_smooth_from_slider(v):
            self.ui.doubleSpinBox_color_smooth.blockSignals(True)
            self.ui.doubleSpinBox_color_smooth.setValue(v * 0.1)
            self.ui.doubleSpinBox_color_smooth.blockSignals(False)
        def _color_trace_smooth_from_spin(val):
            self.ui.slider_color_smooth.blockSignals(True)
            self.ui.slider_color_smooth.setValue(int(round(val * 10)))
            self.ui.slider_color_smooth.blockSignals(False)
        def _color_trace_opt_from_slider(v):
            self.ui.doubleSpinBox_color_opttol.blockSignals(True)
            self.ui.doubleSpinBox_color_opttol.setValue(v * 0.1)
            self.ui.doubleSpinBox_color_opttol.blockSignals(False)
        def _color_trace_opt_from_spin(val):
            self.ui.slider_color_opttol.blockSignals(True)
            self.ui.slider_color_opttol.setValue(int(round(val * 10)))
            self.ui.slider_color_opttol.blockSignals(False)
        
        self.ui.slider_color_smooth.valueChanged.connect(_color_trace_smooth_from_slider)
        self.ui.doubleSpinBox_color_smooth.valueChanged.connect(_color_trace_smooth_from_spin)
        self.ui.slider_color_opttol.valueChanged.connect(_color_trace_opt_from_slider)
        self.ui.doubleSpinBox_color_opttol.valueChanged.connect(_color_trace_opt_from_spin)

        # diffvg优化（与基础页「细节优化」相同联动）
        self.ui.slider_advanced_color_diffvg.valueChanged.connect(self.ui.spinbox_advanced_color_diffvg.setValue)
        self.ui.spinbox_advanced_color_diffvg.valueChanged.connect(self.ui.slider_advanced_color_diffvg.setValue)

        # SuperSVG 参数（整数滑块 ↔ 数值框）
        self.ui.slider_supersvg_stroke_num.valueChanged.connect(self.ui.spinBox_supersvg_stroke_num.setValue)
        self.ui.spinBox_supersvg_stroke_num.valueChanged.connect(self.ui.slider_supersvg_stroke_num.setValue)
        self.ui.slider_supersvg_path_num.valueChanged.connect(self.ui.spinBox_supersvg_path_num.setValue)
        self.ui.spinBox_supersvg_path_num.valueChanged.connect(self.ui.slider_supersvg_path_num.setValue)
        self.ui.slider_supersvg_finetune_iter.valueChanged.connect(self.ui.spinBox_supersvg_finetune_iter.setValue)
        self.ui.spinBox_supersvg_finetune_iter.valueChanged.connect(self.ui.slider_supersvg_finetune_iter.setValue)
        
        # SuperSVG 参数（浮点数滑块 ↔ 数值框）
        def _supersvg_lr_path_from_slider(v):
            self.ui.doubleSpinBox_supersvg_lr_path.blockSignals(True)
            self.ui.doubleSpinBox_supersvg_lr_path.setValue(v * 0.1)
            self.ui.doubleSpinBox_supersvg_lr_path.blockSignals(False)
        def _supersvg_lr_path_from_spin(val):
            self.ui.slider_supersvg_lr_path.blockSignals(True)
            self.ui.slider_supersvg_lr_path.setValue(int(round(val * 10)))
            self.ui.slider_supersvg_lr_path.blockSignals(False)
        def _supersvg_lr_color_from_slider(v):
            self.ui.doubleSpinBox_supersvg_lr_color.blockSignals(True)
            self.ui.doubleSpinBox_supersvg_lr_color.setValue(v * 0.001)
            self.ui.doubleSpinBox_supersvg_lr_color.blockSignals(False)
        def _supersvg_lr_color_from_spin(val):
            self.ui.slider_supersvg_lr_color.blockSignals(True)
            self.ui.slider_supersvg_lr_color.setValue(int(round(val * 1000)))
            self.ui.slider_supersvg_lr_color.blockSignals(False)
        
        self.ui.slider_supersvg_lr_path.valueChanged.connect(_supersvg_lr_path_from_slider)
        self.ui.doubleSpinBox_supersvg_lr_path.valueChanged.connect(_supersvg_lr_path_from_spin)
        self.ui.slider_supersvg_lr_color.valueChanged.connect(_supersvg_lr_color_from_slider)
        self.ui.doubleSpinBox_supersvg_lr_color.valueChanged.connect(_supersvg_lr_color_from_spin)
        

        # 启用拖拽功能
        self.ui.setAcceptDrops(True)

        # 获取当前窗口大小并放大1.3倍
        current_size = self.ui.size()
        new_width = int(current_size.width() * SCALE_FACTOR)
        new_height = int(current_size.height() * SCALE_FACTOR)
        self.ui.resize(new_width, new_height)
        
        # 锁定窗口大小 - 禁止缩放（要解除锁定，请注释掉下面两行）
        self.ui.setFixedSize(new_width, new_height)
        self.ui.setWindowFlags(self.ui.windowFlags() & ~Qt.WindowMaximizeButtonHint)
        
        # 连接UI事件
        self.connect_UI()
        
        self.ui.show()
        
        sys.exit(self.exec_())

    # 初始化参数
    # def parameter_init(self):
    #     self.color_enabled = None
    #     self.color_num = None
    #     self.detail_enabled = None
    #     self.diffvg_iter = None
    #     self.supersvg_enabled = None
    #     self.supersvg_iter = None


# ====================================================== #
# 前后端链接
# ====================================================== #

    # 连接信号槽
    def connect_UI(self):
        # 连接导入图片按钮点击事件
        self.ui.btn_import.clicked.connect(self.import_image)
        self.ui.btn_import_advanced.clicked.connect(self.import_image)

        # 绑定事件处理函数
        self.ui.dragEnterEvent = self.dragEnterEvent
        self.ui.dropEvent = self.dropEvent

        # 连接矢量化按钮点击事件
        self.ui.btn_start.clicked.connect(self.start_vector)
        self.ui.btn_start_advanced.clicked.connect(self.advanced_vector)

        # 连接 SuperSVG 模块的状态变化信号
        self.ui.groupBox_supersvg.toggled.connect(self.on_supersvg_toggled)

        # 高级模式：矢量化方式 ↔ 参数面板
        self.ui.comboBox_advanced_vector_mode.currentIndexChanged.connect(
            self.ui.stackedWidget_advanced_params.setCurrentIndex
        )

    # 开始矢量化处理
    def start_vector(self):
        '''
        开始矢量化处理
        '''

        # 图片存在性检测
        if not self.file_path:
            QMessageBox.warning(self.ui, "错误", "请先导入图片")
            return None

        # 清理残留文件
        potrace_vector.clean_dir(self.output_svg_dir)
        potrace_vector.clean_dir(self.tmp_dir)
        potrace_vector.clean_dir(f'{self.tmp_dir}/refine_svg')

        # 获取用户参数
        # self.color_enabled = self.ui.groupBox_color.isChecked()
        # self.color_num = self.ui.spinbox_color.value()
        # self.detail_enabled = self.ui.groupBox_detail.isChecked()
        # self.diffvg_iter = self.ui.spinbox_detail.value()
        # self.supersvg_enabled = self.ui.groupBox_supersvg.isChecked()
        # self.supersvg_iter = self.ui.spinbox_supersvg.value()
        # print(self.color_enabled, self.color_num, self.detail_enabled, self.diffvg_iter, self.supersvg_enabled, self.supersvg_iter)

        if self.ui.groupBox_supersvg.isChecked():
            self.supersvg2svg(finetune_iter=self.ui.spinbox_supersvg.value())
        else:
            if self.ui.groupBox_color.isChecked():
                self.color_potrace2svg(color_count=self.ui.spinbox_color.value())
            else:
                self.potrace2svg()
            if self.ui.groupBox_detail.isChecked():
                self.diffvg_optmizer(num_iter=self.ui.spinbox_detail.value(), use_lpips_loss=False)
            else:pass

        #清理暂存文件
        # potrace_vector.clean_dir(self.output_svg_dir)
        potrace_vector.clean_dir(self.tmp_dir)
        potrace_vector.clean_dir(f'{self.tmp_dir}/refine_svg')

        self.show_svg()

    # 高级矢量化处理
    def advanced_vector(self):
        '''
        高级矢量化处理
        '''

        # 图片存在性检测
        if not self.file_path:
            QMessageBox.warning(self.ui, "错误", "请先导入图片")
            return None

        # 清理残留文件
        potrace_vector.clean_dir(self.output_svg_dir)
        potrace_vector.clean_dir(self.tmp_dir)
        potrace_vector.clean_dir(f'{self.tmp_dir}/refine_svg')

        path_strategy = ["black", "white", "right", "left", "minority", "majority", "random"]
        color_algo = ["mc", "as"]

        if self.ui.comboBox_advanced_vector_mode.currentIndex() == 0:
            self.potrace2svg(
                t=self.ui.spinbox_potrace_turd.value(),
                a=self.ui.doubleSpinBox_potrace_smooth.value(),
                o=self.ui.doubleSpinBox_potrace_opttolerance.value(),
                z=path_strategy[self.ui.comboBox_potrace_turnpolicy.currentIndex()]
            )
            if self.ui.groupBox_advanced_diffvg.isChecked():
                self.diffvg_optmizer(num_iter=self.ui.spinbox_advanced_diffvg.value(), use_lpips_loss=False)
        elif self.ui.comboBox_advanced_vector_mode.currentIndex() == 1:
            self.color_potrace2svg(
                color_count=self.ui.spinBox_color_count.value(),
                quantization=color_algo[self.ui.comboBox_quant_algo.currentIndex()],
                despeckle=self.ui.spinBox_color_despeckle.value(),
                smoothcorners=self.ui.doubleSpinBox_color_smooth.value(),
                optimizepaths=self.ui.doubleSpinBox_color_opttol.value()
            )
            if self.ui.groupBox_advanced_color_diffvg.isChecked():
                self.diffvg_optmizer(num_iter=self.ui.spinbox_advanced_color_diffvg.value(), use_lpips_loss=False)
        elif self.ui.comboBox_advanced_vector_mode.currentIndex() == 2:
            self.supersvg2svg(
                stroke_num=self.ui.spinBox_supersvg_stroke_num.value(),
                path_num=self.ui.spinBox_supersvg_path_num.value(),
                finetune_iter=self.ui.spinBox_supersvg_finetune_iter.value(),
                lr_path=self.ui.doubleSpinBox_supersvg_lr_path.value(),
                lr_color=self.ui.doubleSpinBox_supersvg_lr_color.value()
            )
        
        #清理暂存文件
        # potrace_vector.clean_dir(self.output_svg_dir)
        potrace_vector.clean_dir(self.tmp_dir)
        potrace_vector.clean_dir(f'{self.tmp_dir}/refine_svg')

        self.show_svg()
        

    # 导入图片按钮点击事件处理函数
    def import_image(self):
        '''
        导入图片按钮点击事件处理函数
        Args:
            self: 应用实例
            
        Returns:
            None
        '''
        try:
            # 打开文件选择对话框
            file_path, _ = QFileDialog.getOpenFileName(
                self.ui, "导入图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp)"
            )

            # 保存文件地址
            self.file_path = file_path
            self.output_svg_path = os.path.join(self.output_svg_dir, 'final_svg.svg')
            
            # 显示图片到预览区域
            pixmap = QPixmap(file_path)
            self.ui.lbl_preview_input.setPixmap(pixmap.scaled(
                self.ui.lbl_preview_input.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            self.ui.lbl_preview_input.setText("")

            return file_path

        except Exception as e:
            QMessageBox.warning(self.ui, "错误", f"导入文件失败: {str(e)}")

    # 显示SVG文件到输出预览按钮点击事件处理函数
    def show_svg(self):
        """
        显示SVG文件到输出预览
        """
        try:
            # 检查是否有导入的图片
            if not self.file_path:
                self.ui.lbl_preview_output.setText("无法显示图片，请先导入")
                return
            
            # 构建SVG文件路径
            svg_path = self.output_svg_path
            
            print(f"[调试] SVG文件路径: {svg_path}")
            print(f"[调试] 输出目录: {self.output_svg_dir}")
            print(f"[调试] 原始文件名: {os.path.basename(self.file_path)}")
            
            # 检查SVG文件是否存在
            if not os.path.exists(svg_path):
                error_msg = f"SVG文件不存在\n路径: {svg_path}"
                print(f"[错误] {error_msg}")
                self.ui.lbl_preview_output.setText(f"SVG文件不存在")
                QMessageBox.warning(self.ui, "错误", error_msg)
                return
            
            # 检查文件大小
            file_size = os.path.getsize(svg_path)
            print(f"[调试] SVG文件大小: {file_size} 字节")
            
            if file_size == 0:
                error_msg = "SVG文件为空"
                print(f"[错误] {error_msg}")
                self.ui.lbl_preview_output.setText(error_msg)
                QMessageBox.warning(self.ui, "错误", error_msg)
                return
            
            # 创建SVG渲染器
            renderer = QSvgRenderer(svg_path)
            
            # 检查渲染器是否有效
            if not renderer.isValid():
                error_msg = f"SVG文件格式无效\n文件: {svg_path}"
                print(f"[错误] {error_msg}")
                self.ui.lbl_preview_output.setText("SVG文件格式无效")
                QMessageBox.warning(self.ui, "错误", error_msg)
                return
            
            # 获取预览标签
            preview_label = self.ui.lbl_preview_output
            print(f"[调试] 预览标签大小: {preview_label.size().width()}x{preview_label.size().height()}")
            
            # 获取SVG默认尺寸
            svg_size = renderer.defaultSize()
            print(f"[调试] SVG默认尺寸: {svg_size.width()}x{svg_size.height()}")
            
            if svg_size.isEmpty():
                svg_size = preview_label.size()
                print(f"[调试] 使用标签尺寸: {svg_size.width()}x{svg_size.height()}")
            
            # 创建与SVG尺寸匹配的Pixmap
            pixmap = QPixmap(svg_size)
            
            # 检查Pixmap是否创建成功
            if pixmap.isNull():
                error_msg = "无法创建Pixmap"
                print(f"[错误] {error_msg}")
                self.ui.lbl_preview_output.setText(error_msg)
                QMessageBox.warning(self.ui, "错误", error_msg)
                return
            
            pixmap.fill(Qt.white)  # 填充白色背景
            
            # 创建QPainter并在Pixmap上渲染SVG
            from PyQt5.QtGui import QPainter
            painter = QPainter(pixmap)
            renderer.render(painter)
            painter.end()
            print(f"[调试] SVG渲染成功")
            
            # 缩放并显示到标签
            scaled_pixmap = pixmap.scaled(
                preview_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            preview_label.setPixmap(scaled_pixmap)
            preview_label.setText("")
            
            print(f"[成功] SVG显示成功")
            
        except Exception as e:
            # 显示详细的错误信息
            import traceback
            error_detail = traceback.format_exc()
            print(f"[异常] {error_detail}")
            
            error_msg = f"加载SVG失败:\n{str(e)}\n\n详细信息:\n{error_detail}"
            self.ui.lbl_preview_output.setText(f"加载SVG失败: {str(e)}")
            QMessageBox.critical(self.ui, "错误", error_msg)

    # 拖拽事件处理函数
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()
    
    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]

        if os.path.exists(files[0]):
            image_path = files[0]
            self.file_path = image_path
            # print(self.file_path)
            # 检查文件是否为图片
            if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                try:
                    if os.path.exists(image_path):
                        self.file_path = image_path
                        # 显示图片到预览区域
                        pixmap = QPixmap(image_path)
                        self.ui.lbl_preview_input.setPixmap(pixmap.scaled(
                            self.ui.lbl_preview_input.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                        ))
                        self.ui.lbl_preview_input.setText("")
                    else:
                        raise FileNotFoundError("文件不存在")
                except Exception as e:
                    QMessageBox.warning(self.ui, "错误", f"导入失败: {str(e)}")

    # SuperSVG 模块状态变化处理
    def on_supersvg_toggled(self, checked):
        '''
        当 SuperSVG 模块被选中时，禁用色彩保留和细节优化模块
        
        Args:
            checked: SuperSVG 模块的选中状态
        '''
        if checked:
            # 当 SuperSVG 被选中时，禁用色彩保留和细节优化模块
            self.ui.groupBox_color.setChecked(False)  # 取消勾选色彩保留
            self.ui.groupBox_color.setEnabled(False)  # 禁用色彩保留模块
            self.ui.groupBox_detail.setChecked(False)  # 取消勾选细节优化
            self.ui.groupBox_detail.setEnabled(False)  # 禁用细节优化模块

        else:
            # 当 SuperSVG 未被选中时，启用色彩保留和细节优化模块
            self.ui.groupBox_color.setEnabled(True)   # 启用色彩保留模块
            self.ui.groupBox_detail.setEnabled(True)   # 启用细节优化模块


# ====================================================== #
# 矢量化功能模块
# ====================================================== #

    # potrace矢量化
    def potrace2svg(self, t=2, a=1, o=0.3, z='minority',r=72):
        '''
        potrace_svg 函数
        Args:
            self: 应用实例
            t: 阈值参数（默认2），控制二值化的阈值，值越小越敏感
            a: 面积参数（默认1），控制输出平滑度，值越大输出越平滑
            o: 曲线优化容差（默认0.3），值越大精度越低
            z: 颜色选择参数（默认'minority'）
            r: 分辨率参数（默认72），控制svg占用显示屏大小，值越大占用越小
        Returns:
            None
        '''
        if not self.file_path:
            QMessageBox.warning(self.ui, "potrace矢量化错误", "请先导入图片")
            return None
        
        try:
            potracerun = potrace_vector.PotraceRunner()
            png_path = potrace_vector.jpg2png(self.file_path, os.path.join(self.tmp_dir, os.path.basename(self.file_path).replace('jpg', 'png')))
            # print(png_path)
            png_path = potrace_vector.convert_4ch_to_3ch(png_path, png_path)
            bmp_path = potrace_vector.png2bmp(png_path, os.path.join(self.tmp_dir, os.path.basename(png_path).replace('.png', '.bmp')))
            # print(bmp_path)
            self.output_svg_path = potracerun.potrace_run(t=t, a=a, o=o, z=z, r=r, input_path=bmp_path, output_path=os.path.join(self.output_svg_dir, 'final_svg.svg'))

            print('potrace矢量化后文件保存路径为' + self.output_svg_path)
            # self.show_svg()

        except Exception as e:
            QMessageBox.warning(self.ui, "错误", f"potrace转换失败: {str(e)}")
            return None

    # 彩色potrace矢量化
    def color_potrace2svg(self, color_count = 0, quantization='mc', dither=None, stack=False, prescale=2, despeckle=2, smoothcorners=1.0, optimizepaths=0.2):
        '''
        color_potrace2svg 函数
        Args:
            self: 应用实例
            color_count: 颜色数量（默认0）
            quantization: 量化算法（默认'mc'）
            dither: 拟色算法（默认True）
            stack: 是否堆栈描摹（默认False）
            prescale: 预缩放比例（默认2）
            despeckle: 抑制斑点（默认2）
            smoothcorners: 平滑转角（默认1.0）
            optimizepaths: 路径优化（默认0.2）
        Returns:
            None
        '''
        if not self.file_path:
            QMessageBox.warning(self.ui, "错误", "请先导入图片")
            return None
        
        try:
            try:
                multiprocessing.freeze_support()
            except Exception as e:
                pass
            print('开始彩色potrace矢量化')
            # print([self.file_path], [self.output_svg_path])
            
            img = io.imread(self.file_path)
            height, width = img.shape[:2]
            # print(height, width)

            color_trace(
                input_list=[self.file_path], 
                output_list=[self.output_svg_path], 
                color_count=color_count, 
                process_count=1, 
                quantization=quantization, 
                dither=dither, 
                stack=stack, 
                prescale=prescale, 
                despeckle=despeckle, 
                smoothcorners=smoothcorners, 
                optimizepaths=optimizepaths,
                background=False,
                width=f"{width/1.25}pt",
                height=f"{height/1.25}pt",
                resolution=None
                )

        except Exception as e:
            QMessageBox.warning(self.ui, "错误", f"彩色potrace转换失败: {str(e)}")
            return None

        print('彩色potrace矢量化后文件保存路径为' + self.output_svg_path)
        # self.show_svg()

    # diffvg优化器
    def diffvg_optmizer(self, num_iter=50, use_lpips_loss=False):
        '''
        diffvg_optmizer 函数
        Args:
            self: 应用实例
            num_iter: 迭代次数（默认50）
            use_lpips_loss: 是否使用lpips损失（默认False）
        Returns:
            None
        '''
        if not self.output_svg_path:
            QMessageBox.warning(self.ui, "错误", "请先矢量化图片")
            return None
        
        try:
            refine_svg(
                self.output_svg_path, 
                self.file_path, 
                use_lpips_loss=use_lpips_loss, 
                num_iter=num_iter)

            output_svg_path = os.path.join(self.tmp_dir, 'refine_svg', f'iter_{num_iter - 1}.svg')
            potrace_vector.clean_dir(self.output_svg_dir)
            move_svg_os(output_svg_path, self.output_svg_dir)
            # self.output_svg_path = os.path.join(self.output_svg_dir, os.path.basename(output_svg_path))


        except Exception as e:
            QMessageBox.warning(self.ui, "错误", f"diffvg优化失败: {str(e)}")
            return None

    # supersvg矢量化
    def supersvg2svg(self, stroke_num=128, path_num=4, finetune_iter=50, lr_path=1.0, lr_color=0.01):
        '''
        supersvg2svg 函数
        Args:
            self: 应用实例
            stroke_num: 笔画数量（默认128）
            path_num: 路径数量（默认4）
            finetune_iter: 微调迭代次数（默认50）
            lr_path: 路径学习率（默认1.0）
            lr_color: 颜色学习率（默认0.01）
        Returns:
            None
        '''
        if not self.file_path:
            QMessageBox.warning(self.ui, "错误", "请先导入图片")
            return None
        
        try:
            bitmap_to_svg(
                input_image_path=self.file_path, 
                output_svg_path=self.output_svg_path, 
                checkpoint_path="./SuperSVG/coarse-model.pt",
                stroke_num=stroke_num, 
                path_num=path_num, 
                finetune_iter=finetune_iter, 
                lr_path=lr_path, 
                lr_color=lr_color,
                verbose=False,
                device="cpu",
                width=224,
                )
        except Exception as e:
            QMessageBox.warning(self.ui, "错误", f"supersvg转换失败: {str(e)}")
            return None

        print('supersvg矢量化后文件保存路径为' + self.output_svg_path)
        # self.show_svg()


if __name__ == '__main__':
    app = App(sys.argv)