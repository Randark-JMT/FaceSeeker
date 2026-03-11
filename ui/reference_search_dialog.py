"""参考库检索对话框 - 展示选定人脸与参考库中相似度最高的 Top 20 便于研判。"""

import os

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.database import DatabaseManager
from core.face_engine import FaceEngine, imread_unicode
from core.reference_match import search_reference_top_k

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _first_image_in_folder(folder_path: str) -> str | None:
    """返回文件夹内第一张支持格式的图片路径，若无则 None。"""
    if not folder_path or not os.path.isdir(folder_path):
        return None
    for f in sorted(os.listdir(folder_path)):
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT:
            p = os.path.join(folder_path, f)
            if os.path.isfile(p):
                return p
    return None


def _cv_to_pixmap(cv_img: np.ndarray, w: int, h: int) -> QPixmap:
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    ih, iw, ch = rgb.shape
    qimg = QImage(rgb.data, iw, ih, ch * iw, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg).scaled(
        w, h,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class ReferenceSearchDialog(QDialog):
    """选定人脸在参考库中检索 Top 20 相似结果，用于研判。"""

    def __init__(self, db: DatabaseManager, face_id: int, parent=None):
        super().__init__(parent)
        self.db = db
        self.face_id = face_id
        self.setWindowTitle("参考库检索 — 人脸 F{}".format(face_id))
        self.setMinimumSize(720, 520)
        self.setModal(False)
        self._build_ui()
        self._load_data()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 16)

        # 选定人脸区域
        self._query_label = QLabel("选定人脸")
        self._query_label.setStyleSheet("font-weight: bold; font-size: 10pt; color: #b0a830;")
        root.addWidget(self._query_label)

        self._query_thumb = QLabel()
        self._query_thumb.setFixedSize(120, 120)
        self._query_thumb.setStyleSheet("border: 2px solid #b0a830; background: #2a2a2a;")
        self._query_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._query_thumb.setText("加载中…")
        root.addWidget(self._query_thumb)

        # 参考库相似度 Top 20
        ref_title = QLabel("参考库相似度 Top 20（从高到低）")
        ref_title.setStyleSheet("font-weight: bold; font-size: 10pt; color: #4caf50; margin-top: 8px;")
        root.addWidget(ref_title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._ref_container = QWidget()
        self._ref_layout = QGridLayout(self._ref_container)
        self._ref_layout.setSpacing(8)
        scroll.setWidget(self._ref_container)
        root.addWidget(scroll)

    def _load_data(self):
        face_row = self.db.get_face_with_file_path(self.face_id)
        if not face_row:
            self._query_thumb.setText("人脸不存在")
            return
        file_path = face_row.get("file_path")
        if not file_path:
            self._query_thumb.setText("无图片路径")
            return
        img = imread_unicode(file_path)
        if img is not None:
            bbox = (face_row["bbox_x"], face_row["bbox_y"], face_row["bbox_w"], face_row["bbox_h"])
            crop = FaceEngine.crop_face(img, bbox)
            pix = _cv_to_pixmap(crop, 120, 120)
            self._query_thumb.setPixmap(pix)
        else:
            self._query_thumb.setText("无法加载图片")

        results = search_reference_top_k(self.db, self.face_id, top_k=20)
        if not results:
            no_ref = QLabel("参考库为空或该人脸无特征，无法检索。")
            no_ref.setStyleSheet("color: #888; padding: 12px;")
            self._ref_layout.addWidget(no_ref, 0, 0)
            return

        for i, (ref, sim) in enumerate(results):
            row, col = i // 5, i % 5
            person_id = ref.get("person_id", "?")
            folder_path = ref.get("folder_path", "")
            thumb = QLabel()
            thumb.setFixedSize(80, 80)
            thumb.setStyleSheet("border: 1px solid #555; background: #333;")
            thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            first_img = _first_image_in_folder(folder_path)
            if first_img:
                img = imread_unicode(first_img)
                if img is not None:
                    # 简单居中裁剪为方形后缩放
                    h, w = img.shape[:2]
                    s = min(h, w)
                    y, x = (h - s) // 2, (w - s) // 2
                    crop = img[y:y + s, x:x + s]
                    pix = _cv_to_pixmap(crop, 80, 80)
                    thumb.setPixmap(pix)
                else:
                    thumb.setText("?")
            else:
                thumb.setText("—")
            sim_label = QLabel(f"{person_id}\n相似度: {sim:.2%}")
            sim_label.setStyleSheet("font-size: 9pt; color: #ccc;")
            sim_label.setWordWrap(True)
            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.addWidget(thumb)
            cell_layout.addWidget(sim_label)
            self._ref_layout.addWidget(cell, row, col)
