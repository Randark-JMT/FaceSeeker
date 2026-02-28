"""清空数据对话框 - 展示数据库状态并选择清空范围。"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from core.database import DatabaseManager


SCOPE_ALL = "all"
SCOPE_LABELED = "labeled"
SCOPE_UNLABELED = "unlabeled"
SCOPE_CLUSTER = "cluster"
SCOPE_REF_MATCH = "ref_match"


class ClearDataDialog(QDialog):
    """点击清空数据后弹出的对话框，展示数据库状态并选择清空范围。"""

    def __init__(self, db: DatabaseManager, parent=None):
        super().__init__(parent)
        self.db = db
        self._scope: str | None = None

        self.setWindowTitle("清空数据")
        self.setMinimumWidth(420)
        self.setModal(True)

        self._build_ui()
        self._refresh_stats()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(16)
        root.setContentsMargins(20, 20, 20, 20)

        # 数据库状态
        stats_group = QGroupBox("数据库当前状态")
        stats_layout = QFormLayout(stats_group)
        self._label_images = QLabel("—")
        self._label_faces = QLabel("—")
        self._label_persons = QLabel("—")
        self._label_labeled = QLabel("—")
        stats_layout.addRow("图片 (images):", self._label_images)
        stats_layout.addRow("人脸 (faces):", self._label_faces)
        stats_layout.addRow("人物 (persons):", self._label_persons)
        stats_layout.addRow("已标记库 (labeled_persons):", self._label_labeled)
        root.addWidget(stats_group)

        # 清空范围
        scope_group = QGroupBox("清空范围")
        scope_group.setStyleSheet(
            """
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
                border-radius: 7px;
                border: 2px solid #888;
                background-color: transparent;
            }
            QRadioButton::indicator:checked {
                background-color: white;
                border: 2px solid white;
            }
            """
        )
        scope_layout = QVBoxLayout(scope_group)
        self._radio_all = QRadioButton("全部 — 清空图片、人脸、人物、已标记库")
        self._radio_labeled = QRadioButton("已标记库 — 清空参考库（labeled_persons）")
        self._radio_unlabeled = QRadioButton("未标记库 — 清空聚类产生的未命名人物及其关联")
        self._radio_cluster = QRadioButton("人脸归类 — 清空所有人脸的人物归属与人物表")
        self._radio_ref_match = QRadioButton("特征库匹配 — 仅解除人脸与参考库/未知的关联")
        self._radio_all.setChecked(True)
        scope_layout.addWidget(self._radio_all)
        scope_layout.addWidget(self._radio_labeled)
        scope_layout.addWidget(self._radio_unlabeled)
        scope_layout.addWidget(self._radio_cluster)
        scope_layout.addWidget(self._radio_ref_match)
        root.addWidget(scope_group)

        # 按钮
        btns = QHBoxLayout()
        btns.addStretch()
        self._clear_btn = QPushButton("确认清空")
        self._clear_btn.setDefault(True)
        self._clear_btn.clicked.connect(self._on_confirm)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(cancel_btn)
        btns.addWidget(self._clear_btn)
        root.addLayout(btns)

    def _refresh_stats(self):
        try:
            stats = self.db.get_table_stats()
            self._label_images.setText(str(stats["images"]))
            self._label_faces.setText(str(stats["faces"]))
            self._label_persons.setText(str(stats["persons"]))
            self._label_labeled.setText(str(stats["labeled_persons"]))
        except Exception as e:
            self._label_images.setText("读取失败")
            self._label_faces.setText(str(e))
            self._label_persons.setText("")
            self._label_labeled.setText("")

    def _on_confirm(self):
        if self._radio_all.isChecked():
            scope = SCOPE_ALL
        elif self._radio_labeled.isChecked():
            scope = SCOPE_LABELED
        elif self._radio_unlabeled.isChecked():
            scope = SCOPE_UNLABELED
        elif self._radio_cluster.isChecked():
            scope = SCOPE_CLUSTER
        elif self._radio_ref_match.isChecked():
            scope = SCOPE_REF_MATCH
        else:
            scope = SCOPE_ALL

        msg = self._get_confirm_message(scope)
        ret = QMessageBox.question(
            self,
            "确认清空",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ret == QMessageBox.StandardButton.Yes:
            self._scope = scope
            self.accept()

    def _get_confirm_message(self, scope: str) -> str:
        if scope == SCOPE_ALL:
            return "确定要清空全部数据吗？将删除图片、人脸、人物、已标记库。此操作不可撤销。"
        if scope == SCOPE_LABELED:
            return "确定要清空已标记库吗？参考库中的大头照数据将被删除，人物匹配需重新导入参考库。"
        if scope == SCOPE_UNLABELED:
            return "确定要清空未标记库吗？将删除聚类产生的未命名人物，解除这些人脸的人物归属。"
        if scope == SCOPE_CLUSTER:
            return "确定要清空人脸归类结果吗？所有人脸的人物归属将被清除，人物表将清空。图片和已标记库保留。"
        if scope == SCOPE_REF_MATCH:
            return "确定要清除特征库匹配结果吗？将解除人脸与参考库/未知人物的关联，可重新执行参考库匹配。"
        return "确定要执行清空吗？"

    def get_scope(self) -> str | None:
        """返回用户选择的清空范围。"""
        return self._scope
