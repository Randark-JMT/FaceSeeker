"""数据目录选择对话框 - 在主窗口显示前询问用户数据文件存储位置"""

import os

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QRadioButton, QLineEdit, QPushButton, QFileDialog,
    QGroupBox, QButtonGroup, QFrame,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from core.config import Config
from ui import APP_NAME, APP_VERSION


def _format_size(size_bytes: int) -> str:
    """将字节数格式化为可读字符串"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


class DataDirDialog(QDialog):
    """数据目录选择对话框

    首次启动时弹出，让用户选择数据文件（数据库、日志）的存储位置：
    - 使用默认目录（AppData）
    - 手动指定目录
    如果指定目录中已有数据文件，会提示用户将加载已有数据。
    """

    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self._selected_dir: str | None = None  # None = 使用默认

        self.setWindowTitle(APP_NAME)
        self.setFixedWidth(520)

        self._build_ui()
        self._apply_style()

        # 初始化状态
        if config.is_data_dir_configured and config._data_dir:
            self._radio_custom.setChecked(True)
            self._path_edit.setText(config._data_dir)
        else:
            self._radio_default.setChecked(True)

        self._on_mode_changed()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        # ---- 标题 ----
        title = QLabel("选择数据存储位置")
        title_font = QFont()
        title_font.setPointSize(13)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        desc = QLabel(
            "请选择数据库和日志文件的存储目录。\n"
            "模型等资源文件将自动管理，无需手动设置。"
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #aaa; margin-bottom: 4px;")
        layout.addWidget(desc)

        # ---- 选项组 ----
        group = QGroupBox()
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(8)

        self._btn_group = QButtonGroup(self)

        # 默认目录选项
        self._radio_default = QRadioButton("使用默认目录")
        self._btn_group.addButton(self._radio_default)
        group_layout.addWidget(self._radio_default)

        default_path_label = QLabel(f"  📂  {self.config.default_data_dir}")
        default_path_label.setStyleSheet("color: #888; font-size: 9pt; margin-left: 20px;")
        default_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        group_layout.addWidget(default_path_label)

        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #444;")
        group_layout.addWidget(line)

        # 自定义目录选项
        self._radio_custom = QRadioButton("自定义目录")
        self._btn_group.addButton(self._radio_custom)
        group_layout.addWidget(self._radio_custom)

        # 路径输入行
        path_row = QHBoxLayout()
        path_row.setContentsMargins(20, 0, 0, 0)
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText('点击"浏览"选择目录...')
        self._path_edit.setReadOnly(True)
        self._browse_btn = QPushButton("浏览...")
        self._browse_btn.setFixedWidth(72)
        path_row.addWidget(self._path_edit)
        path_row.addWidget(self._browse_btn)
        group_layout.addLayout(path_row)

        layout.addWidget(group)

        # ---- 数据文件检测状态 ----
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("font-size: 9pt; padding: 6px;")
        self._status_label.setVisible(False)
        layout.addWidget(self._status_label)

        # ---- 按钮 ----
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._ok_btn = QPushButton("确定")
        self._ok_btn.setFixedWidth(90)
        self._ok_btn.setDefault(True)
        btn_row.addWidget(self._ok_btn)
        layout.addLayout(btn_row)

        # ---- 信号 ----
        self._radio_default.toggled.connect(self._on_mode_changed)
        self._radio_custom.toggled.connect(self._on_mode_changed)
        self._browse_btn.clicked.connect(self._on_browse)
        self._ok_btn.clicked.connect(self._on_ok)

    def _apply_style(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
            QGroupBox {
                background-color: #252526;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 12px;
                margin-top: 4px;
            }
            QRadioButton {
                font-size: 10pt;
                spacing: 6px;
            }
            QRadioButton::indicator {
                width: 14px; height: 14px;
            }
            QLineEdit {
                background-color: #3c3c3c;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 4px 8px;
                color: #d4d4d4;
            }
            QPushButton {
                background-color: #0e639c;
                border: none;
                border-radius: 3px;
                padding: 5px 14px;
                color: #fff;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:pressed {
                background-color: #0d5689;
            }
        """)

    # ---- 事件处理 ----

    def _on_mode_changed(self):
        """切换默认/自定义模式"""
        is_custom = self._radio_custom.isChecked()
        self._path_edit.setEnabled(is_custom)
        self._browse_btn.setEnabled(is_custom)

        if is_custom and self._path_edit.text().strip():
            self._check_existing_data(self._path_edit.text().strip())
        elif not is_custom:
            self._check_existing_data(self.config.default_data_dir)

    def _on_browse(self):
        """浏览选择目录"""
        start_dir = self._path_edit.text().strip() or self.config.default_data_dir
        chosen = QFileDialog.getExistingDirectory(
            self, "选择数据目录", start_dir,
            QFileDialog.Option.ShowDirsOnly
        )
        if chosen:
            self._path_edit.setText(chosen)
            self._check_existing_data(chosen)

    def _check_existing_data(self, directory: str):
        """检查目录下是否存在已有数据文件并更新状态显示"""
        if not directory or not os.path.isdir(directory):
            self._status_label.setVisible(False)
            return

        info = Config.check_existing_data(directory)
        if info["db"] or info["log"]:
            parts = []
            if info["db"]:
                parts.append(f"数据库 ({_format_size(info['db_size'])})")
            if info["log"]:
                parts.append(f"日志 ({_format_size(info['log_size'])})")
            self._status_label.setText(
                f"✅  在该目录中发现已有数据文件：{'、'.join(parts)}\n"
                f"    将直接加载已有数据，不会丢失。"
            )
            self._status_label.setStyleSheet(
                "font-size: 9pt; padding: 6px; color: #4ec9b0; "
                "background-color: #1a3a2a; border-radius: 3px;"
            )
            self._status_label.setVisible(True)
        else:
            self._status_label.setText(
                "📁  该目录中没有已有数据文件，将创建新的数据库。"
            )
            self._status_label.setStyleSheet(
                "font-size: 9pt; padding: 6px; color: #888; "
                "background-color: #2a2a2a; border-radius: 3px;"
            )
            self._status_label.setVisible(True)

    def _on_ok(self):
        """确认按钮"""
        if self._radio_custom.isChecked():
            path = self._path_edit.text().strip()
            if not path:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "提示", "请选择一个目录")
                return
            self._selected_dir = path
        else:
            self._selected_dir = None  # 使用默认

        self.accept()

    # ---- 结果接口 ----

    def get_selected_dir(self) -> str | None:
        """获取用户选择的目录路径。None 表示使用默认 AppData 目录。"""
        return self._selected_dir


def show_data_dir_dialog(config: Config) -> bool:
    """显示数据目录选择对话框并将结果应用到配置

    仅在首次启动（未配置过）或数据目录失效时调用。

    Args:
        config: 配置实例

    Returns:
        True 表示用户确认了选择，False 表示用户关闭了对话框（取消）
    """
    dlg = DataDirDialog(config)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return False

    config.set_data_dir(dlg.get_selected_dir())
    return True
