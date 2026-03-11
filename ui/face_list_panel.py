"""当前图片的人脸列表面板（性能优化版）

优化：优先从缩略图缓存加载，避免每次都从完整图像裁剪。
"""

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QFrame,
    QMenu,
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, Signal

from core.face_engine import FaceEngine
from core.thumb_cache import ThumbCache


class FaceCard(QFrame):
    """单个人脸卡片，右键菜单可「在参考库中进行检索」"""

    search_in_reference_requested = Signal(int)  # face_id

    def __init__(self, pix: QPixmap, index: int, score: float,
                 person_id: int | None = None, blur_score: float = 0.0,
                 face_id: int | None = None,
                 parent=None):
        super().__init__(parent)
        self._face_id = face_id
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "FaceCard { background: #2d2d2d; border: 1px solid #444; border-radius: 4px; }"
        )
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        if face_id is not None:
            self.setToolTip("右键：在参考库中检索")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # 缩略图
        thumb_label = QLabel()
        thumb_label.setFixedSize(64, 64)
        thumb_label.setPixmap(pix)
        thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(thumb_label)

        # 信息
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)
        info_layout.addWidget(QLabel(f"人脸 #{index}"))
        info_layout.addWidget(QLabel(f"置信度: {score:.3f}"))
        person_text = f"人物: P{person_id}" if person_id is not None else "人物: 未归类"
        info_layout.addWidget(QLabel(person_text))

        # 清晰度分数（颜色编码，0-100 评分）
        blur_label = QLabel(f"清晰度: {blur_score:.1f}")
        if blur_score >= 40:
            blur_label.setStyleSheet("color: #4caf50;")  # 绿色 - 清晰
        elif blur_score >= 15:
            blur_label.setStyleSheet("color: #ff9800;")  # 橙色 - 模糊但可辨认
        else:
            blur_label.setStyleSheet("color: #f44336;")  # 红色 - 严重模糊
        info_layout.addWidget(blur_label)

        layout.addLayout(info_layout)
        layout.addStretch()

    def contextMenuEvent(self, event):
        if self._face_id is None:
            return
        menu = QMenu(self)
        menu.addAction(
            "在参考库中进行检索",
            lambda: self.search_in_reference_requested.emit(self._face_id),
        )
        menu.exec(event.globalPos())


def _cv_to_pixmap(cv_img: np.ndarray, w: int, h: int) -> QPixmap:
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    ih, iw, ch = rgb.shape
    qimg = QImage(rgb.data, iw, ih, ch * iw, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg).scaled(
        w, h,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class FaceListPanel(QWidget):
    """显示当前选中图片的所有人脸"""

    face_selected = Signal(int)  # 发射人脸索引
    search_in_reference_requested = Signal(int)  # face_id — 在参考库中检索该人脸

    def __init__(self, thumb_cache: ThumbCache | None = None, parent=None):
        super().__init__(parent)
        self._thumb_cache = thumb_cache

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("人脸列表")
        title.setStyleSheet("font-weight: bold; font-size: 11pt; padding: 4px;")
        layout.addWidget(title)

        # 滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._container_layout.setSpacing(4)
        scroll.setWidget(self._container)
        layout.addWidget(scroll)

    def update_faces(self, cv_image: np.ndarray | None, faces_data: list):
        """
        更新人脸列表。

        Args:
            cv_image: 原始 BGR 图像
            faces_data: [{bbox, score, person_id}, ...] 从数据库行转换
        """
        # 清空
        while self._container_layout.count():
            item = self._container_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

        if cv_image is None or not faces_data:
            placeholder = QLabel("无人脸数据")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("color: #888; padding: 20px;")
            self._container_layout.addWidget(placeholder)
            return

        for idx, fd in enumerate(faces_data):
            face_id = fd.get("id")
            bbox = (fd["bbox_x"], fd["bbox_y"], fd["bbox_w"], fd["bbox_h"])

            # ★ 优先从缩略图缓存加载
            pix = None
            if self._thumb_cache and face_id:
                pix = self._thumb_cache.get_pixmap(face_id, 64, 64)

            if pix is None:
                # 兜底：从完整图像裁剪
                thumb = FaceEngine.crop_face(cv_image, bbox)
                pix = _cv_to_pixmap(thumb, 64, 64)

            card = FaceCard(pix, idx, fd["score"], fd.get("person_id"),
                           blur_score=fd.get("blur_score", 0.0),
                           face_id=face_id)
            card.search_in_reference_requested.connect(self.search_in_reference_requested)
            self._container_layout.addWidget(card)
