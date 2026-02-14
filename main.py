#!/usr/bin/env python3
"""FaceSeeker - PySide6 可视化人脸识别系统"""

import sys
import os

from PySide6.QtWidgets import QApplication, QMessageBox

from core.config import get_config
from core.logger import setup_logger, get_logger, console
from core.database import DatabaseManager
from core.face_engine import FaceEngine
from ui import APP_NAME, APP_VERSION
from ui.main_window import MainWindow
from ui.data_dir_dialog import show_data_dir_dialog


def main():
    # 初始化配置
    config = get_config()

    app = QApplication(sys.argv)

    # 全局暗色样式
    app.setStyleSheet("""
        QWidget {
            background-color: #1e1e1e;
            color: #d4d4d4;
            font-size: 13px;
        }
        QToolBar {
            background-color: #2d2d2d;
            border-bottom: 1px solid #444;
            spacing: 6px;
            padding: 4px;
        }
        QToolBar QToolButton {
            background-color: #3a3a3a;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 4px 10px;
            color: #d4d4d4;
        }
        QToolBar QToolButton:hover {
            background-color: #4a4a4a;
        }
        QListWidget {
            background-color: #252526;
            border: 1px solid #333;
        }
        QListWidget::item:selected {
            background-color: #094771;
        }
        QScrollArea {
            border: none;
        }
    """)

    # ---- 首次启动或数据目录失效时，弹出数据目录选择对话框 ----
    need_dialog = False
    if not config.is_data_dir_configured:
        # 从未配置过：首次启动
        need_dialog = True
    elif config._data_dir and not os.path.isdir(config._data_dir):
        # 之前配置的自定义目录已不存在
        need_dialog = True

    if need_dialog:
        if not show_data_dir_dialog(config):
            # 用户关闭了对话框，直接退出程序
            sys.exit(0)

    # 初始化日志系统（必须在数据目录确定之后）
    logger = setup_logger(config.log_path)
    logger.info("=" * 60)
    logger.info("FaceSeeker 启动")
    logger.info(f"数据目录: {config.data_dir}")
    logger.info(f"数据库路径: {config.database_path}")
    logger.info(f"日志路径: {config.log_path}")
    logger.info("=" * 60)
    # 获取模型文件路径（只读资源，从资源目录读取）
    detection_model = config.get_resource_path(os.path.join("models", "face_detection_yunet_2023mar.onnx"))
    recognition_model = config.get_resource_path(os.path.join("models", "face_recognition_sface_2021dec.onnx"))
    
    # 检查模型文件
    missing = []
    if not os.path.exists(detection_model):
        missing.append(
            f"检测模型: {detection_model}\n"
            "  下载地址: https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet"
        )
        logger.error(f"缺少检测模型: {detection_model}")
    if not os.path.exists(recognition_model):
        missing.append(
            f"识别模型: {recognition_model}\n"
            "  下载地址: https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface"
        )
        logger.error(f"缺少识别模型: {recognition_model}")

    if missing:
        logger.critical("模型文件缺失，程序无法启动")
        QMessageBox.critical(
            None, "缺少模型文件",
            "请将以下模型文件放入 models/ 目录:\n\n" + "\n\n".join(missing),
        )
        sys.exit(1)

    # 初始化核心组件
    try:
        logger.info("初始化数据库...")
        db = DatabaseManager(config.database_path)
        logger.info("初始化人脸识别引擎...")
        engine = FaceEngine(detection_model, recognition_model)
        logger.info(f"人脸识别引擎初始化成功 [后端: {engine.backend_name}]")
    except Exception as e:
        logger.critical(f"初始化失败: {e}", exc_info=True)
        QMessageBox.critical(
            None, "初始化失败",
            f"程序初始化失败:\n{str(e)}\n\n请查看日志文件: {config.log_path}",
        )
        sys.exit(1)

    window = MainWindow(engine, db, config)
    window.setWindowTitle(f"{APP_NAME} - {APP_VERSION}  [后端: {engine.backend_name}]")
    window.show()
    
    logger.info("主窗口已显示，程序就绪")

    ret = app.exec()
    logger.info("程序正在退出...")
    db.close()
    logger.info("数据库已关闭")
    logger.info("FaceSeeker 已退出")
    sys.exit(ret)


if __name__ == "__main__":
    main()
