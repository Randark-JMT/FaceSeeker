#!/usr/bin/env python3
"""FaceSeeker - PySide6 可视化人脸识别系统"""

import sys
import os

from PySide6.QtWidgets import QApplication, QMessageBox

from core.database import DatabaseManager
from core.face_engine import FaceEngine
from ui.main_window import MainWindow


def get_app_dir():
    """获取程序所在目录（源代码运行时为main.py目录，打包后为exe目录）"""
    if getattr(sys, 'frozen', False):
        # 打包后运行：返回exe所在目录
        return os.path.dirname(sys.executable)
    else:
        # 源代码运行：返回main.py所在目录
        return os.path.dirname(os.path.abspath(__file__))


def get_resource_path(relative_path):
    """获取资源文件路径（打包后从临时目录读取，源代码运行从项目目录读取）"""
    if getattr(sys, 'frozen', False):
        # 打包后：资源文件在PyInstaller的临时目录
        base_path = sys._MEIPASS
    else:
        # 源代码运行：资源文件在项目目录
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


# 模型文件路径（只读资源，从资源目录读取）
DETECTION_MODEL = get_resource_path(os.path.join("models", "face_detection_yunet_2023mar.onnx"))
RECOGNITION_MODEL = get_resource_path(os.path.join("models", "face_recognition_sface_2021dec.onnx"))

# 数据库文件路径（用户数据，始终放在程序所在目录）
DB_PATH = os.path.join(get_app_dir(), "faceseeker.db")


def main():
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
        QStatusBar {
            background-color: #007acc;
            color: white;
        }
        QProgressDialog {
            background-color: #2d2d2d;
        }
        QSplitter::handle {
            background-color: #333;
        }
    """)

    # 检查模型文件
    missing = []
    if not os.path.exists(DETECTION_MODEL):
        missing.append(
            f"检测模型: {DETECTION_MODEL}\n"
            "  下载地址: https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet"
        )
    if not os.path.exists(RECOGNITION_MODEL):
        missing.append(
            f"识别模型: {RECOGNITION_MODEL}\n"
            "  下载地址: https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface"
        )

    if missing:
        QMessageBox.critical(
            None, "缺少模型文件",
            "请将以下模型文件放入 models/ 目录:\n\n" + "\n\n".join(missing),
        )
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
