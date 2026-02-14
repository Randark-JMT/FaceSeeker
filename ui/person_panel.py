"""人物归类面板 - 按人物分组展示人脸（性能优化版）

优化要点：
1. FlowLayout：缩略图自动换行，避免水平方向元素挤压
2. 异步缩略图加载：后台线程生成缩略图，UI 不卡顿
3. 懒加载：只在滚动到可见区域附近时才创建 PersonGroup
4. 缩略图缓存：磁盘缓存 + 内存 LRU，避免重复读取完整图片
"""

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QFrame, QLineEdit, QPushButton, QComboBox,
    QLayout, QSizePolicy,
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, Signal, QRect, QSize, QPoint, QTimer

from core.database import DatabaseManager
from core.face_engine import FaceEngine, imread_unicode
from core.thumb_cache import ThumbCache, ThumbLoaderWorker

# 每个人物分组初始显示的缩略图数量
MAX_THUMBS_PER_GROUP = 8
# 滚动懒加载：每次加载的人物组数量
LAZY_BATCH_SIZE = 15


# ---- FlowLayout：自动换行的缩略图布局 ----

class FlowLayout(QLayout):
    """流式布局：子控件自动换行排列，避免水平挤压"""

    def __init__(self, parent=None, margin=0, spacing=4):
        super().__init__(parent)
        self._items: list = []
        self._h_spacing = spacing
        self._v_spacing = spacing
        if margin >= 0:
            self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        size += QSize(m.left() + m.right(), m.top() + m.bottom())
        return size

    def _do_layout(self, rect, test_only=False) -> int:
        m = self.contentsMargins()
        effective = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x = effective.x()
        y = effective.y()
        line_height = 0

        for item in self._items:
            w = item.sizeHint().width()
            h = item.sizeHint().height()
            next_x = x + w + self._h_spacing
            if next_x - self._h_spacing > effective.right() and line_height > 0:
                x = effective.x()
                y += line_height + self._v_spacing
                line_height = 0
                next_x = x + w + self._h_spacing
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))
            x = next_x
            line_height = max(line_height, h)

        return y + line_height - rect.y() + m.bottom()


# ---- 可点击的缩略图和标签 ----

class ClickableThumb(QLabel):
    """可双击的人脸缩略图"""

    double_clicked = Signal(int)  # image_id

    def __init__(self, image_id: int, face_id: int = 0, parent=None):
        super().__init__(parent)
        self._image_id = image_id
        self.face_id = face_id
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit(self._image_id)


class ClickableMoreLabel(QLabel):
    """可双击的 "+N" 展开标签"""

    double_clicked = Signal()

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setFixedSize(56, 56)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            "border: 1px solid #555; color: #4a9eff; font-size: 11pt; font-weight: bold;"
        )
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("双击加载更多（每次10张）")

    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit()


# ---- 单个人物分组 ----

class PersonGroup(QFrame):
    """单个人物的折叠分组（使用 FlowLayout + 异步缩略图）"""

    name_changed = Signal(int, str)   # person_id, new_name
    face_double_clicked = Signal(int) # image_id

    def __init__(self, person_id: int, name: str, face_rows: list,
                 total_face_count: int, thumb_cache: ThumbCache | None = None,
                 parent=None):
        super().__init__(parent)
        self.person_id = person_id
        self.face_rows = face_rows
        self._total_face_count = total_face_count
        self._shown_count = min(MAX_THUMBS_PER_GROUP, len(face_rows))
        self._thumb_cache = thumb_cache
        # face_id -> ClickableThumb 映射，用于异步更新
        self._thumb_widgets: dict[int, ClickableThumb] = {}

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "PersonGroup { background: #2a2a2a; border: 1px solid #444; "
            "border-radius: 4px; margin: 2px; }"
        )

        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(8, 8, 8, 8)
        self._main_layout.setSpacing(6)

        # 标题行
        header = QHBoxLayout()
        person_id_label = QLabel(f"P{person_id}")
        person_id_label.setStyleSheet("font-weight: bold; color: #4a9eff; font-size: 10pt;")
        person_id_label.setFixedWidth(40)
        header.addWidget(person_id_label)

        self._name_edit = QLineEdit(name)
        self._name_edit.setFixedWidth(120)
        self._name_edit.setStyleSheet("background: #3a3a3a; border: 1px solid #555; padding: 2px 4px;")
        self._name_edit.editingFinished.connect(self._on_name_changed)
        header.addWidget(self._name_edit)
        header.addWidget(QLabel(f"({total_face_count} 张人脸)"))
        header.addStretch()
        self._main_layout.addLayout(header)

        # 缩略图容器（FlowLayout 自动换行）
        self.thumb_container = QWidget()
        self.thumb_flow = FlowLayout(self.thumb_container, margin=0, spacing=4)
        self._main_layout.addWidget(self.thumb_container)

        # 初始创建占位缩略图
        self._build_thumbs()

    def _build_thumbs(self):
        """构建缩略图控件（仅创建占位符，实际图像由缓存/异步加载）"""
        # 清空
        while self.thumb_flow.count():
            item = self.thumb_flow.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        self._thumb_widgets.clear()

        shown = self.face_rows[:self._shown_count]
        for row in shown:
            face_id = row["id"]
            thumb = ClickableThumb(row["image_id"], face_id)
            thumb.setFixedSize(56, 56)
            thumb.setStyleSheet("border: 1px solid #555; background: #333;")
            thumb.setToolTip(f"双击跳转 | 来源: {row['filename']}")

            # 尝试从缓存加载
            if self._thumb_cache:
                pix = self._thumb_cache.get_pixmap(face_id)
                if pix is not None:
                    thumb.setPixmap(pix)
                else:
                    thumb.setText("...")
                    thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            else:
                # 没有缓存系统时，使用传统同步方式（兜底）
                self._load_thumb_sync(thumb, row)

            thumb.double_clicked.connect(self.face_double_clicked)
            self.thumb_flow.addWidget(thumb)
            self._thumb_widgets[face_id] = thumb

        # +N 标签
        overflow = self._total_face_count - self._shown_count
        if overflow > 0:
            more_label = ClickableMoreLabel(f"+{overflow}")
            more_label.double_clicked.connect(self._on_show_more)
            self.thumb_flow.addWidget(more_label)

    @staticmethod
    def _load_thumb_sync(thumb: ClickableThumb, row):
        """同步加载缩略图（兜底方案）"""
        img = imread_unicode(row["file_path"])
        if img is not None:
            bbox = (row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"])
            crop = FaceEngine.crop_face(img, bbox)
            pix = _cv_to_pixmap(crop, 56, 56)
            thumb.setPixmap(pix)
        else:
            thumb.setText("?")
            thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_thumb_pixmap(self, face_id: int, pix: QPixmap):
        """异步加载完成后，设置指定人脸的缩略图"""
        thumb = self._thumb_widgets.get(face_id)
        if thumb is not None:
            thumb.setPixmap(pix)

    def get_uncached_requests(self) -> list[tuple]:
        """返回需要异步加载的缩略图请求列表: [(face_id, file_path, bbox), ...]"""
        requests = []
        for row in self.face_rows[:self._shown_count]:
            face_id = row["id"]
            if self._thumb_cache and self._thumb_cache.has_cache(face_id):
                continue
            bbox = (row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"])
            requests.append((face_id, row["file_path"], bbox))
        return requests

    def _on_show_more(self):
        """双击 +号，展开更多人脸（需要从 DB 补充数据）"""
        old_count = self._shown_count
        self._shown_count = min(self._shown_count + 10, self._total_face_count)
        # 如果当前 face_rows 不够，需要外部补充
        if self._shown_count > len(self.face_rows):
            # 信号通知外部补充数据（PersonPanel 处理）
            pass
        self._build_thumbs()
        # 触发异步加载新增的缩略图
        self._request_async_load()

    def _request_async_load(self):
        """请求异步加载未缓存的缩略图（由 PersonPanel 协调）"""
        # 通过查找父级 PersonPanel 来触发异步加载
        panel = self.parent()
        while panel is not None:
            if isinstance(panel, PersonPanel):
                panel._load_thumbs_for_group(self)
                break
            panel = panel.parent()

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


# ---- 人物归类面板（主面板） ----

class PersonPanel(QWidget):
    """人物归类面板（支持懒加载 + 异步缩略图）"""

    navigate_to_image = Signal(int)  # image_id — 外部连接此信号实现跳转

    def __init__(self, db: DatabaseManager, thumb_cache: ThumbCache | None = None, parent=None):
        super().__init__(parent)
        self.db = db
        self._thumb_cache = thumb_cache

        # 排序状态
        self._sort_by = "id"   # "id" 或 "count"
        self._sort_order = "asc"

        # 懒加载状态
        self._all_persons: list = []    # 排序后的完整人物列表
        self._loaded_count = 0          # 已加载的 PersonGroup 数量
        self._person_groups: list[PersonGroup] = []

        # 异步缩略图加载器
        self._thumb_worker: ThumbLoaderWorker | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 标题栏 + 排序控件
        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)

        title = QLabel("人物归类")
        title.setStyleSheet("font-weight: bold; font-size: 11pt; padding: 4px;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        sort_label = QLabel("排序:")
        sort_label.setStyleSheet("font-size: 9pt;")
        header_layout.addWidget(sort_label)

        self._sort_combo = QComboBox()
        self._sort_combo.addItem("编号", "id")
        self._sort_combo.addItem("出现次数", "count")
        self._sort_combo.setStyleSheet(
            "QComboBox { background: #2a2a2a; border: 1px solid #555; padding: 2px 6px; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView { background: #2a2a2a; border: 1px solid #555; }"
        )
        self._sort_combo.currentIndexChanged.connect(self._on_sort_changed)
        header_layout.addWidget(self._sort_combo)

        self._order_btn = QPushButton("↑")
        self._order_btn.setFixedSize(30, 24)
        self._order_btn.setToolTip("切换升序/降序")
        self._order_btn.setStyleSheet(
            "QPushButton { background: #2a2a2a; border: 1px solid #555; font-size: 12pt; }"
            "QPushButton:hover { background: #3a3a3a; }"
        )
        self._order_btn.clicked.connect(self._toggle_sort_order)
        header_layout.addWidget(self._order_btn)

        layout.addLayout(header_layout)

        # 滚动区域
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._container_layout.setSpacing(4)
        self._scroll.setWidget(self._container)
        layout.addWidget(self._scroll)

        # 监听滚动以触发懒加载
        self._scroll.verticalScrollBar().valueChanged.connect(self._on_scroll)

    # ---- 刷新与懒加载 ----

    def refresh(self):
        """从数据库重新加载人物分组（懒加载版本）"""
        # 停止正在进行的异步加载
        self._cancel_thumb_worker()

        # 清空
        self._clear_groups()

        persons = self.db.get_all_persons()
        if not persons:
            placeholder = QLabel("尚未进行人脸归类")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("color: #888; padding: 20px;")
            self._container_layout.addWidget(placeholder)
            self._all_persons = []
            return

        # 排序
        self._all_persons = self._sort_persons(persons)
        self._loaded_count = 0
        self._person_groups = []

        # 首批加载
        self._load_next_batch()

    def _load_next_batch(self):
        """加载下一批 PersonGroup（懒加载核心）"""
        if self._loaded_count >= len(self._all_persons):
            return

        batch_end = min(self._loaded_count + LAZY_BATCH_SIZE, len(self._all_persons))
        all_requests = []  # 收集所有需要异步加载的缩略图请求

        for i in range(self._loaded_count, batch_end):
            person = self._all_persons[i]
            # 只查询需要显示的人脸（而非全部）
            face_rows = self.db.get_faces_by_person(person["id"], limit=MAX_THUMBS_PER_GROUP)
            if not face_rows:
                continue

            total_count = person["face_count"]
            group = PersonGroup(
                person["id"], person["name"], face_rows,
                total_count, self._thumb_cache,
            )
            group.name_changed.connect(self._on_name_changed)
            group.face_double_clicked.connect(self.navigate_to_image)
            self._container_layout.addWidget(group)
            self._person_groups.append(group)

            # 收集异步请求
            all_requests.extend(group.get_uncached_requests())

        self._loaded_count = batch_end

        # 启动异步缩略图加载
        if all_requests and self._thumb_cache:
            self._start_thumb_worker(all_requests)

    def _on_scroll(self, value):
        """滚动到底部附近时加载更多"""
        sb = self._scroll.verticalScrollBar()
        if sb.maximum() == 0:
            return
        # 当滚动到 80% 以下时，触发下一批加载
        if value >= sb.maximum() * 0.8:
            if self._loaded_count < len(self._all_persons):
                self._load_next_batch()

    def _clear_groups(self):
        """清空所有 PersonGroup"""
        self._person_groups.clear()
        while self._container_layout.count():
            item = self._container_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

    # ---- 异步缩略图加载 ----

    def _start_thumb_worker(self, requests: list[tuple]):
        """启动后台缩略图加载线程"""
        self._cancel_thumb_worker()
        worker = ThumbLoaderWorker(self._thumb_cache, requests, self)
        worker.thumb_ready.connect(self._on_thumb_ready)
        worker.batch_done.connect(self._on_thumb_batch_done)
        worker.finished.connect(worker.deleteLater)
        self._thumb_worker = worker
        worker.start()

    def _cancel_thumb_worker(self):
        """取消正在进行的异步加载"""
        if self._thumb_worker is not None:
            self._thumb_worker.cancel()
            self._thumb_worker.thumb_ready.disconnect(self._on_thumb_ready)
            self._thumb_worker.batch_done.disconnect(self._on_thumb_batch_done)
            self._thumb_worker = None

    def _on_thumb_ready(self, face_id: int, pix: QPixmap):
        """异步缩略图加载完成，更新对应控件"""
        for group in self._person_groups:
            group.set_thumb_pixmap(face_id, pix)

    def _on_thumb_batch_done(self):
        """一批缩略图全部加载完成"""
        self._thumb_worker = None

    def _load_thumbs_for_group(self, group: PersonGroup):
        """为特定 PersonGroup 启动异步缩略图加载"""
        requests = group.get_uncached_requests()
        if requests and self._thumb_cache:
            self._start_thumb_worker(requests)

    # ---- 排序与事件 ----

    def _on_name_changed(self, person_id: int, new_name: str):
        if new_name:
            self.db.update_person_name(person_id, new_name)

    def _on_sort_changed(self):
        self._sort_by = self._sort_combo.currentData()
        self.refresh()

    def _toggle_sort_order(self):
        if self._sort_order == "asc":
            self._sort_order = "desc"
            self._order_btn.setText("↓")
        else:
            self._sort_order = "asc"
            self._order_btn.setText("↑")
        self.refresh()

    def _sort_persons(self, persons: list) -> list:
        if self._sort_by == "id":
            key_func = lambda p: p["id"]
        else:
            key_func = lambda p: p["face_count"]
        reverse = (self._sort_order == "desc")
        return sorted(persons, key=key_func, reverse=reverse)
