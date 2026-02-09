"""人物归类面板 - 按人物分组展示人脸"""

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QFrame, QLineEdit, QApplication,
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, Signal

from core.database import DatabaseManager
from core.face_engine import FaceEngine, imread_unicode

# 每个人物分组最多显示的缩略图数量
MAX_THUMBS_PER_GROUP = 8


class PersonGroup(QFrame):
    """单个人物的折叠分组"""

    name_changed = Signal(int, str)  # person_id, new_name

    def __init__(self, person_id: int, name: str, face_rows: list, parent=None):
        super().__init__(parent)
        self.person_id = person_id
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "PersonGroup { background: #2a2a2a; border: 1px solid #444; "
            "border-radius: 4px; margin: 2px; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # 标题行：名称编辑 + 人脸数
        header = QHBoxLayout()
        self._name_edit = QLineEdit(name)
        self._name_edit.setFixedWidth(120)
        self._name_edit.setStyleSheet("background: #3a3a3a; border: 1px solid #555; padding: 2px 4px;")
        self._name_edit.editingFinished.connect(self._on_name_changed)
        header.addWidget(self._name_edit)
        header.addWidget(QLabel(f"({len(face_rows)} 张人脸)"))
        header.addStretch()
        layout.addLayout(header)

        # 人脸缩略图（限制数量，避免阻塞主线程）
        thumb_layout = QHBoxLayout()
        thumb_layout.setSpacing(4)
        shown = face_rows[:MAX_THUMBS_PER_GROUP]
        for row in shown:
            thumb_label = self._make_thumb(row)
            thumb_layout.addWidget(thumb_label)

        overflow = len(face_rows) - len(shown)
        if overflow > 0:
            more_label = QLabel(f"+{overflow}")
            more_label.setFixedSize(56, 56)
            more_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            more_label.setStyleSheet(
                "border: 1px solid #555; color: #aaa; font-size: 14px;"
            )
            thumb_layout.addWidget(more_label)

        thumb_layout.addStretch()
        layout.addLayout(thumb_layout)

    @staticmethod
    def _make_thumb(row) -> QLabel:
        thumb_label = QLabel()
        thumb_label.setFixedSize(56, 56)
        thumb_label.setStyleSheet("border: 1px solid #555;")
        thumb_label.setToolTip(f"来源: {row['filename']}")

        img = imread_unicode(row["file_path"])
        if img is not None:
            bbox = (row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"])
            crop = FaceEngine.crop_face(img, bbox)
            pix = _cv_to_pixmap(crop, 56, 56)
            thumb_label.setPixmap(pix)
        else:
            thumb_label.setText("?")
            thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return thumb_label

    def _on_name_changed(self):
        self.name_changed.emit(self.person_id, self._name_edit.text().strip())


def _cv_to_pixmap(cv_img: np.ndarray, w: int, h: int) -> QPixmap:
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    ih, iw, ch = rgb.shape
    qimg = QImage(rgb.data, iw, ih, ch * iw, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg).scaled(
        w, h,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class PersonPanel(QWidget):
    """人物归类面板"""

    def __init__(self, db: DatabaseManager, parent=None):
        super().__init__(parent)
        self.db = db

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("人物归类")
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 4px;")
        layout.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._container_layout.setSpacing(4)
        scroll.setWidget(self._container)
        layout.addWidget(scroll)

    def refresh(self):
        """从数据库重新加载人物分组（逐个添加，保持 UI 响应）"""
        # 清空
        while self._container_layout.count():
            item = self._container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        persons = self.db.get_all_persons()
        if not persons:
            placeholder = QLabel("尚未进行人脸归类")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("color: #888; padding: 20px;")
            self._container_layout.addWidget(placeholder)
            return

        for i, person in enumerate(persons):
            face_rows = self.db.get_faces_by_person(person["id"])
            if not face_rows:
                continue
            group = PersonGroup(person["id"], person["name"], face_rows)
            group.name_changed.connect(self._on_name_changed)
            self._container_layout.addWidget(group)
            # 每添加几个分组就让 Qt 事件循环处理一次，避免长时间冻结
            if (i + 1) % 3 == 0:
                QApplication.processEvents()

    def _on_name_changed(self, person_id: int, new_name: str):
        if new_name:
            self.db.update_person_name(person_id, new_name)
