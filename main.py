#!/usr/bin/env python3
"""FaceSeeker - PySide6 可视化人脸识别系统"""

import sys
import os

from PySide6.QtWidgets import QApplication, QMessageBox

from core.config import get_config
from core.logger import setup_logger, get_logger, console
from core.database import DatabaseManager
from core.face_engine import FaceEngine
from ui.main_window import MainWindow


def main():
    # 初始化配置
    config = get_config()
    
    # 初始化日志系统
    logger = setup_logger(config.log_path)
    logger.info("=" * 60)
    logger.info("FaceSeeker 启动")
    logger.info(f"缓存目录: {config.cache_dir}")
    logger.info(f"数据库路径: {config.database_path}")
    logger.info(f"日志路径: {config.log_path}")
    logger.info("=" * 60)
    
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
      获取模型文件路径（只读资源，从资源目录读取）
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
    window.setWindowTitle(f"FaceSeeker - 人脸识别系统  [后端: {engine.backend_name}]")
    window.show()
    
    logger.info("主窗口已显示，程序就绪")

    ret = app.exec()
    logger.info("程序正在退出...")
    db.close()
    logger.info("数据库已关闭")
    logger.info("FaceSeeker 已退出"
        sys.exit(1)

    # 初始化核心组件
    db = DatabaseManager(DB_PATH)
    engine = FaceEngine(DETECTION_MODEL, RECOGNITION_MODEL)

    window = MainWindow(engine, db)
    window.setWindowTitle(f"FaceSeeker - 人脸识别系统  [后端: {engine.backend_name}]")
    window.show()

    ret = app.exec()
    db.close()
    sys.exit(ret)


if __name__ == "__main__":
    main()
