"""主窗口 - 组装所有 UI 部件并协调业务逻辑（性能优化版 v2）

优化要点：
1. 导入与识别分离：导入仅注册文件元数据（极快），识别仅处理未分析图片
2. DetectWorker 有界并发：同时只有 N 个 future 在内存中，杜绝 OOM
3. Worker 线程内裁剪缩略图，只返回极小 JPEG 字节，不返回完整图像
4. 图片列表区分已分析/未分析状态
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from itertools import islice

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QSplitter,
    QToolBar, QListWidget, QListWidgetItem, QMessageBox,
    QFileDialog, QProgressDialog, QStatusBar, QLabel, QSlider,
    QApplication,
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QAction

from ui import APP_NAME, APP_VERSION
from core.config import Config
from core.logger import get_logger, log_opencv_error
from core.database import DatabaseManager
from core.face_engine import FaceEngine, FaceData, imread_unicode
from core.face_cluster import FaceCluster
from core.labeled_import import LabeledImportWorker, load_labeled_dataset
from core.reference_match import ReferenceMatchWorker
from core.thumb_cache import ThumbCache
from ui.image_viewer import ImageViewer
from ui.face_list_panel import FaceListPanel
from ui.person_panel import PersonPanel


SUPPORTED_FORMATS = "图片文件 (*.jpg *.jpeg *.png *.bmp *.tiff *.webp)"
IMAGE_LIST_PAGE_SIZE = 500


# ---- 后台工作线程 ----

class DetectWorker(QThread):
    """后台人脸检测线程（有界并发 + 内存安全）
    
    核心改进：
    - 使用滑动窗口提交 future，同时最多 INFLIGHT_LIMIT 个图像在内存中
    - worker 线程内完成裁剪+JPEG 编码，只传回几 KB 的缩略图字节
    - 不再返回完整图像，从根本上解决万张图片 OOM 问题
    """

    progress = Signal(int, int, str)  # current, total, filename
    finished_all = Signal(int)        # 实际处理的图片数
    error = Signal(str)

    def __init__(self, engine: FaceEngine, db: DatabaseManager,
                 image_items: list[tuple[int, str]],
                 thumb_cache: ThumbCache | None = None,
                 num_workers: int | None = None,
                 blur_threshold: float = 0.0):
        """
        Args:
            image_items: [(image_id, file_path), ...] 待检测的图片列表
            blur_threshold: 模糊度阈值，低于此值的人脸将被丢弃。0 表示不过滤。
        """
        super().__init__()
        self.logger = get_logger()
        self.engine = engine
        self.db = db
        self.image_items = image_items
        self.thumb_cache = thumb_cache
        self.blur_threshold = blur_threshold
        if num_workers is None:
            if engine.backend_name == "CUDA":
                # 多线程向 GPU 提交任务，提高利用率（单线程时 GPU 常处于等待状态）
                self.num_workers = min(os.cpu_count() or 4, 6)
            else:
                self.num_workers = max(os.cpu_count() or 4, 4)
        else:
            self.num_workers = num_workers
        self.logger.info(f"DetectWorker: {len(image_items)} 张待检测，线程数: {self.num_workers}，模糊阈值: {self.blur_threshold}")

    def _process_one(self, engine: FaceEngine, fpath: str):
        """在 worker 线程中处理单张图片，返回轻量结果（不含完整图像）"""
        image = imread_unicode(fpath)
        if image is None:
            return None

        h, w = image.shape[:2]
        faces = engine.detect(image)

        # ★ 模糊度检测：在特征提取之前过滤，节省计算量
        if self.blur_threshold > 0:
            filtered = []
            for face in faces:
                crop = FaceEngine.crop_face(image, face.bbox)
                face.blur_score = FaceEngine.compute_blur_score(crop)
                if face.blur_score >= self.blur_threshold:
                    filtered.append(face)
            faces = filtered
        else:
            # 即使不过滤也计算分数，用于 UI 显示
            for face in faces:
                crop = FaceEngine.crop_face(image, face.bbox)
                face.blur_score = FaceEngine.compute_blur_score(crop)

        engine.extract_features(image, faces)

        # ★ 在 worker 线程中裁剪+编码缩略图（几 KB），不传回完整图像（几 MB）
        thumb_data: list[bytes | None] = []
        if self.thumb_cache:
            for face in faces:
                encoded = ThumbCache.encode_crop(image, face.bbox)
                thumb_data.append(encoded)

        del image  # ★ 立即释放完整图像内存
        return w, h, faces, thumb_data

    def run(self):
        total = len(self.image_items)
        if total == 0:
            self.finished_all.emit(0)
            return

        # 为每个 worker 线程准备独立的 engine
        _local = threading.local()
        engines = [self.engine.clone() for _ in range(self.num_workers)]
        _engine_idx = [0]
        _idx_lock = threading.Lock()

        def _get_thread_engine() -> FaceEngine:
            if not hasattr(_local, 'engine'):
                with _idx_lock:
                    idx = _engine_idx[0]
                    _engine_idx[0] += 1
                _local.engine = engines[idx]
            return _local.engine

        def worker(fpath: str):
            engine = _get_thread_engine()
            return self._process_one(engine, fpath)

        # ★ 有界并发：同时最多 INFLIGHT_LIMIT 个 future，杜绝 OOM
        INFLIGHT_LIMIT = max(self.num_workers * 3, 6)
        BATCH_COMMIT_SIZE = 50

        processed = 0
        uncommitted = 0
        self.db.begin()

        try:
            with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                item_iter = iter(self.image_items)
                active: dict = {}  # future -> (image_id, fpath)

                # 初始填充
                for image_id, fpath in islice(item_iter, INFLIGHT_LIMIT):
                    fut = pool.submit(worker, fpath)
                    active[fut] = (image_id, fpath)

                while active:
                    # 等待至少一个 future 完成
                    done, _ = wait(active.keys(), return_when=FIRST_COMPLETED)

                    for fut in done:
                        image_id, fpath = active.pop(fut)
                        processed += 1
                        self.progress.emit(processed, total, os.path.basename(fpath))

                        try:
                            result = fut.result()
                            if result is None:
                                # 图片无法读取，标记为已分析（0 人脸）避免反复重试
                                self.db.mark_image_analyzed(image_id, 0, 0, 0)
                            else:
                                w, h, faces, thumb_data = result

                                # 写入人脸数据
                                for i, face in enumerate(faces):
                                    face_id = self.db.add_face(
                                        image_id, face.bbox, face.landmarks,
                                        face.score, face.feature,
                                        blur_score=face.blur_score,
                                    )
                                    # 写入预编码的缩略图
                                    if self.thumb_cache and i < len(thumb_data) and thumb_data[i]:
                                        self.thumb_cache.save_from_bytes(face_id, thumb_data[i])

                                self.db.mark_image_analyzed(image_id, w, h, len(faces))

                            uncommitted += 1
                            if uncommitted >= BATCH_COMMIT_SIZE:
                                self.db.commit()
                                self.db.begin()
                                uncommitted = 0

                        except Exception as e:
                            log_opencv_error("DetectWorker", e, suppress=True)
                            self.error.emit(f"{os.path.basename(fpath)}: {e}")

                        # ★ 补充一个新 future，维持滑动窗口
                        try:
                            next_id, next_path = next(item_iter)
                            new_fut = pool.submit(worker, next_path)
                            active[new_fut] = (next_id, next_path)
                        except StopIteration:
                            pass

            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

        self.logger.info(f"DetectWorker 完成: 处理了 {processed} 张图片")
        self.finished_all.emit(processed)


class ClusterWorker(QThread):
    """后台聚类线程"""

    progress = Signal(int, int, str)
    finished_cluster = Signal(dict)

    def __init__(self, cluster: FaceCluster, threshold: float = 0.363, incremental: bool = True):
        super().__init__()
        self.cluster_engine = cluster
        self.threshold = threshold
        self.incremental = incremental

    def run(self):
        result = self.cluster_engine.cluster(
            self.threshold,
            progress_cb=lambda cur, tot, txt: self.progress.emit(cur, tot, txt),
            incremental=self.incremental,
        )
        self.finished_cluster.emit(result)


# ---- 主窗口 ----

class MainWindow(QMainWindow):

    def __init__(self, engine: FaceEngine, db: DatabaseManager, config: Config):
        super().__init__()
        self.engine = engine
        self.db = db
        self.config = config
        self.cluster_engine = FaceCluster(db)
        self.logger = get_logger()

        self._current_image_id: int | None = None
        self._worker: QThread | None = None

        # 缩略图缓存
        self._thumb_cache = ThumbCache(config.thumb_cache_dir)

        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.resize(1440, 800)

        self._build_toolbar()
        self._build_ui()
        self._build_statusbar()

        # 延迟加载
        QTimer.singleShot(0, self._load_image_list)

        self.logger.info("主窗口初始化完成")

    # ---- 构建 UI ----

    def _build_toolbar(self):
        tb = QToolBar("工具栏")
        tb.setMovable(False)
        self.addToolBar(tb)

        self._act_import_images = QAction("导入图片", self)
        self._act_import_images.triggered.connect(self._on_import_images)
        tb.addAction(self._act_import_images)

        self._act_import_folder = QAction("导入文件夹", self)
        self._act_import_folder.triggered.connect(self._on_import_folder)
        tb.addAction(self._act_import_folder)

        tb.addSeparator()

        self._act_detect = QAction("开始识别", self)
        self._act_detect.setToolTip("对未识别的图片进行人脸检测（已识别的不会重复处理）")
        self._act_detect.triggered.connect(self._on_detect)
        tb.addAction(self._act_detect)

        self._act_redetect = QAction("重新识别当前图片", self)
        self._act_redetect.setToolTip("删除当前图片的旧识别结果，重新进行人脸检测")
        self._act_redetect.triggered.connect(self._on_redetect_current)
        tb.addAction(self._act_redetect)

        self._act_cluster = QAction("人脸归类", self)
        self._act_cluster.triggered.connect(self._on_cluster)
        tb.addAction(self._act_cluster)

        tb.addSeparator()

        self._act_import_reference = QAction("导入参考库", self)
        self._act_import_reference.setToolTip(
            "选择根目录（含人物子文件夹，文件夹名=人物编号，内为大头照）\n"
            "对每人多张大头照进行人脸检测+特征综合，写入参考库"
        )
        self._act_import_reference.triggered.connect(self._on_import_reference)
        tb.addAction(self._act_import_reference)

        self._act_reference_match = QAction("参考库匹配", self)
        self._act_reference_match.setToolTip(
            "将未归类人脸与参考库进行相似度匹配\n"
            "匹配成功则标记人物，未匹配则标记为「未知」"
        )
        self._act_reference_match.triggered.connect(self._on_reference_match)
        tb.addAction(self._act_reference_match)

        tb.addSeparator()

        self._act_clear = QAction("清空数据", self)
        self._act_clear.triggered.connect(self._on_clear_all)
        tb.addAction(self._act_clear)

        tb.addSeparator()

        self._act_switch_db = QAction("切换数据库连接", self)
        self._act_switch_db.triggered.connect(self._on_switch_database)
        tb.addAction(self._act_switch_db)

        tb.addSeparator()

        # 聚类阈值滑块
        thresh_label = QLabel("聚类阈值:")
        thresh_label.setStyleSheet("padding: 0 4px;")
        tb.addWidget(thresh_label)
        self._thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self._thresh_slider.setRange(50, 100)
        self._thresh_slider.setValue(80)
        self._thresh_slider.setFixedWidth(120)
        self._thresh_slider.setToolTip("调整人脸聚类的余弦相似度阈值\n值越高→分类越严格，人物组越多\n值越低→分类越宽松，更容易合并")
        tb.addWidget(self._thresh_slider)
        self._thresh_value_label = QLabel("0.80")
        self._thresh_value_label.setFixedWidth(36)
        self._thresh_value_label.setStyleSheet("padding: 0 4px;")
        tb.addWidget(self._thresh_value_label)
        self._thresh_slider.valueChanged.connect(
            lambda v: self._thresh_value_label.setText(f"{v / 100:.2f}")
        )

        tb.addSeparator()

        # 模糊度阈值滑块
        blur_label = QLabel("清晰度阈值:")
        blur_label.setStyleSheet("padding: 0 4px;")
        tb.addWidget(blur_label)
        self._blur_slider = QSlider(Qt.Orientation.Horizontal)
        self._blur_slider.setRange(0, 100)
        self._blur_slider.setValue(0)
        self._blur_slider.setFixedWidth(120)
        self._blur_slider.setToolTip(
            "人脸清晰度过滤阈值（多指标融合评分 0-100）\n"
            "低于此值的模糊人脸将被丢弃\n"
            "0 = 不过滤（保留所有人脸）\n"
            "参考：< 15 严重模糊  |  15-40 模糊但可辨认  |  > 40 清晰\n"
            "算法：Laplacian + Tenengrad(Sobel) + FFT高频占比 加权融合"
        )
        tb.addWidget(self._blur_slider)
        self._blur_value_label = QLabel("关闭")
        self._blur_value_label.setFixedWidth(36)
        self._blur_value_label.setStyleSheet("padding: 0 4px;")
        tb.addWidget(self._blur_value_label)
        self._blur_slider.valueChanged.connect(self._on_blur_slider_changed)

        tb.addSeparator()

        # 参考库匹配阈值
        ref_label = QLabel("参考库阈值:")
        ref_label.setStyleSheet("padding: 0 4px;")
        tb.addWidget(ref_label)
        self._ref_thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self._ref_thresh_slider.setRange(50, 100)
        self._ref_thresh_slider.setValue(60)
        self._ref_thresh_slider.setFixedWidth(100)
        self._ref_thresh_slider.setToolTip("参考库匹配的余弦相似度阈值，高于此值则标记为该人物")
        tb.addWidget(self._ref_thresh_slider)
        self._ref_thresh_value_label = QLabel("0.60")
        self._ref_thresh_value_label.setFixedWidth(36)
        self._ref_thresh_value_label.setStyleSheet("padding: 0 4px;")
        tb.addWidget(self._ref_thresh_value_label)
        self._ref_thresh_slider.valueChanged.connect(
            lambda v: self._ref_thresh_value_label.setText(f"{v / 100:.2f}")
        )

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(4, 4, 4, 4)

        # 上半部分：图像查看器
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._original_viewer = ImageViewer("原始图像")
        self._result_viewer = ImageViewer("识别结果")
        top_splitter.addWidget(self._original_viewer)
        top_splitter.addWidget(self._result_viewer)
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 1)

        # 下半部分：列表与面板
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)

        # 图片列表
        image_list_container = QWidget()
        il_layout = QVBoxLayout(image_list_container)
        il_layout.setContentsMargins(0, 0, 0, 0)
        il_label = QLabel("图片列表")
        il_label.setStyleSheet("font-weight: bold; font-size: 11pt; padding: 4px;")
        il_layout.addWidget(il_label)
        self._image_list = QListWidget()
        self._image_list.currentItemChanged.connect(self._on_image_selected)
        il_layout.addWidget(self._image_list)
        bottom_splitter.addWidget(image_list_container)

        # 人脸列表
        self._face_panel = FaceListPanel(thumb_cache=self._thumb_cache)
        bottom_splitter.addWidget(self._face_panel)

        # 人物归类
        self._person_panel = PersonPanel(self.db, thumb_cache=self._thumb_cache)
        self._person_panel.navigate_to_image.connect(self._navigate_to_image)
        bottom_splitter.addWidget(self._person_panel)

        # 默认宽度：图片列表 500px，人脸列表与人物归类约 1:1（各 470px，总 1440）
        bottom_splitter.setSizes([500, 470, 470])
        bottom_splitter.setStretchFactor(0, 1)   # 图片列表不随窗口拉伸
        bottom_splitter.setStretchFactor(1, 1)   # 人脸列表
        bottom_splitter.setStretchFactor(2, 1)   # 人物归类（与人脸同比例）

        # 主分割器
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(top_splitter)
        main_splitter.addWidget(bottom_splitter)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 2)

        root_layout.addWidget(main_splitter)

    def _build_statusbar(self):
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("就绪")

    # ---- 数据加载 ----

    def _load_image_list(self):
        """从数据库加载图片列表"""
        self._image_list.clear()

        offset = 0
        total_loaded = 0
        while True:
            rows = self.db.get_images_paginated(offset, IMAGE_LIST_PAGE_SIZE)
            if not rows:
                break
            for row in rows:
                display_name = row['filename']
                if row['relative_path']:
                    display_name = f"{row['relative_path']}/{row['filename']}"

                # ★ 区分已分析/未分析状态
                if row['analyzed']:
                    label = f"{display_name}  [{row['face_count']} 人脸]"
                else:
                    label = f"{display_name}  [未识别]"

                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, row["id"])
                # 未分析的项用不同颜色
                if not row['analyzed']:
                    item.setForeground(Qt.GlobalColor.darkYellow)
                self._image_list.addItem(item)

            total_loaded += len(rows)
            offset += IMAGE_LIST_PAGE_SIZE
            if total_loaded % 2000 == 0:
                QApplication.processEvents()

        # 加载人物归类
        self._person_panel.refresh()

        # ★ 状态栏显示统计
        self._update_statusbar_stats()

    def _update_statusbar_stats(self):
        """更新状态栏的统计信息"""
        total = self.db.get_image_count()
        analyzed = self.db.get_analyzed_count()
        unanalyzed = total - analyzed
        if total == 0:
            self._statusbar.showMessage("就绪 - 请导入图片")
        elif unanalyzed > 0:
            self._statusbar.showMessage(
                f"共 {total} 张图片 | 已识别 {analyzed} 张 | 待识别 {unanalyzed} 张"
            )
        else:
            face_count = self.db.get_face_count()
            self._statusbar.showMessage(
                f"共 {total} 张图片 | 全部已识别 | {face_count} 张人脸"
            )

    def _show_image(self, image_id: int):
        """显示选中图片的原始图 + 识别结果图"""
        self._current_image_id = image_id
        img_row = self.db.get_image(image_id)
        if img_row is None:
            return

        cv_img = imread_unicode(img_row["file_path"])
        if cv_img is None:
            self._statusbar.showMessage(f"无法读取图片: {img_row['file_path']}")
            return

        self._original_viewer.set_image(cv_img)

        # 如果未分析，只显示原图
        if not img_row["analyzed"]:
            self._result_viewer.set_image(cv_img)
            self._face_panel.update_faces(None, [])
            return

        face_rows = self.db.get_faces_by_image(image_id)
        faces_data = [dict(row) for row in face_rows]

        if faces_data:
            face_datas = []
            person_ids = []
            for fd in faces_data:
                landmarks = DatabaseManager.landmarks_from_json(fd["landmarks"])
                face_datas.append(FaceData(
                    bbox=(fd["bbox_x"], fd["bbox_y"], fd["bbox_w"], fd["bbox_h"]),
                    landmarks=landmarks,
                    score=fd["score"],
                ))
                person_ids.append(fd["person_id"])

            annotated = self.engine.visualize(cv_img, face_datas, person_ids)
            self._result_viewer.set_image(annotated)
        else:
            self._result_viewer.set_image(cv_img)

        self._face_panel.update_faces(cv_img, faces_data)

    def _navigate_to_image(self, image_id: int):
        for i in range(self._image_list.count()):
            item = self._image_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == image_id:
                self._image_list.setCurrentItem(item)
                return

    # ---- 工具栏回调 ----

    def _on_blur_slider_changed(self, value: int):
        self._blur_value_label.setText("关闭" if value == 0 else str(value))

    def _on_image_selected(self, current: QListWidgetItem, _previous=None):
        if current is None:
            return
        image_id = current.data(Qt.ItemDataRole.UserRole)
        self._show_image(image_id)

    def _on_import_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "", SUPPORTED_FORMATS
        )
        if paths:
            self.logger.info(f"用户选择导入 {len(paths)} 张图片")
            self._do_import(paths)

    def _on_import_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if not folder:
            return
        self.logger.info(f"用户选择导入文件夹: {folder}")
        exts = FaceEngine.SUPPORTED_FORMATS
        paths = []
        for root, dirs, files in os.walk(folder):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    paths.append(os.path.join(root, f))
        if not paths:
            QMessageBox.information(self, "提示", "该文件夹及其子文件夹下没有找到支持的图片格式。")
            return
        self.logger.info(f"扫描到 {len(paths)} 张图片")
        self._do_import(paths, folder)

    # ---- ★ 导入流程：仅注册文件元数据，极快 ----

    def _do_import(self, paths: list[str], base_folder: str = ""):
        """将图片路径注册到数据库（不做人脸识别），自动跳过已存在的"""
        # 批量获取已有路径，避免逐条查询
        existing = self.db.get_existing_paths()
        new_paths = [p for p in paths if p not in existing]
        skipped = len(paths) - len(new_paths)

        if not new_paths:
            QMessageBox.information(
                self, "导入完成",
                f"所有 {len(paths)} 张图片已在数据库中，无需重复导入。"
            )
            return

        self.logger.info(f"导入: 新增 {len(new_paths)} 张，跳过 {skipped} 张已存在")

        # 批量事务注册（纯数据库操作，极快）
        self.db.begin()
        try:
            for fpath in new_paths:
                filename = os.path.basename(fpath)
                relative_path = ""
                if base_folder:
                    try:
                        relative_path = os.path.relpath(os.path.dirname(fpath), base_folder)
                        if relative_path == ".":
                            relative_path = ""
                    except ValueError:
                        relative_path = ""
                self.db.add_image(fpath, filename, relative_path=relative_path, analyzed=0)
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

        # 刷新列表
        self._load_image_list()

        # 提示用户
        unanalyzed = self.db.get_unanalyzed_count()
        msg = f"成功导入 {len(new_paths)} 张新图片"
        if skipped > 0:
            msg += f"（跳过 {skipped} 张已存在）"
        msg += f"\n\n当前共 {unanalyzed} 张图片待识别，点击「开始识别」进行人脸检测。"
        QMessageBox.information(self, "导入完成", msg)

    # ---- ★ 识别流程：仅处理未分析的图片 ----

    def _on_detect(self):
        """对未分析的图片进行人脸检测"""
        unanalyzed = self.db.get_unanalyzed_images()
        if not unanalyzed:
            QMessageBox.information(self, "提示", "所有图片已完成识别，无需重复处理。")
            return

        count = len(unanalyzed)
        self.logger.info(f"开始识别 {count} 张未分析图片")

        self._statusbar.showMessage(f"正在识别 {count} 张图片...")
        self._set_actions_enabled(False)

        self._progress = QProgressDialog(f"正在识别 {count} 张图片...", "取消", 0, count, self)
        self._progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress.setMinimumDuration(0)

        # 构建 (image_id, file_path) 列表
        items = [(row["id"], row["file_path"]) for row in unanalyzed]

        blur_threshold = self._blur_slider.value()
        worker = DetectWorker(
            self.engine, self.db, items,
            thumb_cache=self._thumb_cache,
            blur_threshold=float(blur_threshold),
        )
        worker.progress.connect(self._on_detect_progress)
        worker.finished_all.connect(self._on_detect_done)
        worker.error.connect(lambda msg: self.logger.warning(f"检测错误: {msg}"))
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_detect_progress(self, current: int, total: int, filename: str):
        self._progress.setLabelText(f"识别 [{current}/{total}]: {filename}")
        self._progress.setValue(current)

    def _on_detect_done(self, processed_count: int):
        self._progress.close()
        self._load_image_list()
        self._set_actions_enabled(True)

        if self._image_list.count() > 0 and self._current_image_id is None:
            self._image_list.setCurrentRow(0)

    # ---- 重新识别当前图片 ----

    def _on_redetect_current(self):
        """删除当前图片的旧识别结果，重新进行人脸检测"""
        if self._current_image_id is None:
            QMessageBox.information(self, "提示", "请先选择一张图片。")
            return

        image_id = self._current_image_id
        img_row = self.db.get_image(image_id)
        if img_row is None:
            QMessageBox.warning(self, "错误", "找不到该图片记录。")
            return

        fpath = img_row["file_path"]
        self._statusbar.showMessage(f"正在重新识别: {os.path.basename(fpath)}...")
        QApplication.processEvents()

        # 读取图像
        cv_img = imread_unicode(fpath)
        if cv_img is None:
            QMessageBox.warning(self, "错误", f"无法读取图片:\n{fpath}")
            return

        # 删除旧的人脸数据及缩略图缓存
        old_face_ids = self.db.delete_faces_by_image(image_id)
        for fid in old_face_ids:
            self._thumb_cache.invalidate(fid)

        # 人脸检测
        h, w = cv_img.shape[:2]
        faces = self.engine.detect(cv_img)

        # 模糊度检测与过滤
        blur_threshold = self._blur_slider.value()
        discarded = 0
        for face in faces:
            crop = FaceEngine.crop_face(cv_img, face.bbox)
            face.blur_score = FaceEngine.compute_blur_score(crop)

        if blur_threshold > 0:
            before_count = len(faces)
            faces = [f for f in faces if f.blur_score >= blur_threshold]
            discarded = before_count - len(faces)

        # 特征提取（仅对通过模糊度过滤的人脸）
        self.engine.extract_features(cv_img, faces)

        # 保存到数据库
        for face in faces:
            face_id = self.db.add_face(
                image_id, face.bbox, face.landmarks,
                face.score, face.feature,
                blur_score=face.blur_score,
            )
            # 生成缩略图缓存
            if self._thumb_cache:
                self._thumb_cache.save_from_image(face_id, cv_img, face.bbox)

        # 标记为已分析
        self.db.mark_image_analyzed(image_id, w, h, len(faces))

        # 刷新显示
        self._show_image(image_id)

        # 更新图片列表中的当前条目
        current_item = self._image_list.currentItem()
        if current_item:
            display_name = img_row['filename']
            if img_row['relative_path']:
                display_name = f"{img_row['relative_path']}/{img_row['filename']}"
            label = f"{display_name}  [{len(faces)} 人脸]"
            current_item.setText(label)
            current_item.setForeground(Qt.GlobalColor.white)

        msg = f"重新识别完成: 检测到 {len(faces)} 张人脸"
        if discarded > 0:
            msg += f"（已过滤 {discarded} 张模糊人脸）"
        self._statusbar.showMessage(msg)
        self.logger.info(f"重新识别图片 {image_id}: {len(faces)} 张人脸, 过滤 {discarded} 张模糊")
        self._update_statusbar_stats()

    # ---- 聚类 ----

    def _on_cluster(self):
        face_count = self.db.get_face_count()
        if face_count == 0:
            QMessageBox.information(self, "提示", "没有可用的人脸特征数据，请先导入并识别图片。")
            return

        threshold = self._thresh_slider.value() / 100.0
        self.logger.info(f"开始人脸聚类（全量），阈值={threshold:.2f}，共 {face_count} 张人脸")

        self._statusbar.showMessage("正在进行人脸归类...")
        self._set_actions_enabled(False)

        self._cluster_progress = QProgressDialog(
            f"正在归类 {face_count} 张人脸（阈值 {threshold:.2f}）...", None, 0, 100, self
        )
        self._cluster_progress.setWindowTitle("人脸归类")
        self._cluster_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._cluster_progress.setMinimumDuration(0)
        self._cluster_progress.setValue(0)

        # 全量模式：每次清空旧归类结果后重新聚类，方便调试不同阈值
        worker = ClusterWorker(self.cluster_engine, threshold, incremental=False)
        worker.progress.connect(self._on_cluster_progress)
        worker.finished_cluster.connect(self._on_cluster_done)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_cluster_progress(self, current: int, total: int, text: str):
        self._cluster_progress.setMaximum(max(total, 1))
        self._cluster_progress.setValue(current)
        self._cluster_progress.setLabelText(text)

    def _on_cluster_done(self, result: dict):
        self._cluster_progress.close()
        self._set_actions_enabled(True)
        person_count = len(result)
        face_count = sum(len(v) for v in result.values())
        self.logger.info(f"聚类完成: {face_count} 张人脸归为 {person_count} 个人物")
        self._statusbar.showMessage(f"归类完成: {face_count} 张人脸归为 {person_count} 个人物")
        self._person_panel.refresh()

        if self._current_image_id is not None:
            self._show_image(self._current_image_id)

    # ---- 参考库导入与匹配 ----

    def _on_import_reference(self):
        """导入已标记数据集：选择根目录，遍历人物子文件夹，多图特征综合写入 labeled_persons"""
        folder = QFileDialog.getExistingDirectory(self, "选择参考库根目录")
        if not folder:
            return
        items = list(load_labeled_dataset(folder))
        if not items:
            QMessageBox.warning(
                self, "导入参考库",
                "该目录下没有有效的人物子文件夹，或子文件夹内无支持的图片格式。\n"
                "格式要求：根目录/人物编号(文件夹名)/大头照图片"
            )
            return

        self.logger.info(f"导入参考库: 选择根目录 {folder}，共 {len(items)} 个人物")
        self._statusbar.showMessage("正在导入参考库...")
        self._set_actions_enabled(False)

        self._ref_progress = QProgressDialog(
            f"正在导入参考库（共 {len(items)} 人）...", "取消", 0, len(items), self
        )
        self._ref_progress.setWindowTitle("导入参考库")
        self._ref_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._ref_progress.setMinimumDuration(0)
        self._ref_progress.setValue(0)

        worker = LabeledImportWorker(self.engine, self.db, folder)
        worker.progress.connect(self._on_ref_import_progress)
        worker.finished_all.connect(self._on_ref_import_done)
        worker.error.connect(lambda msg: self.logger.warning(f"参考库导入: {msg}"))
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        self._ref_progress.canceled.connect(worker.cancel)
        worker.start()

    def _on_ref_import_progress(self, current: int, total: int, person_id: str):
        self._ref_progress.setLabelText(f"导入 [{current}/{total}]: {person_id}")
        self._ref_progress.setValue(current)

    def _on_ref_import_done(self, imported_count: int):
        self._ref_progress.close()
        self._set_actions_enabled(True)
        self._statusbar.showMessage(f"参考库导入完成: 成功 {imported_count} 人")
        self.logger.info(f"参考库导入完成: {imported_count} 人")

    def _on_reference_match(self):
        """将未归类人脸与参考库相似度匹配，匹配则标记人物，未匹配则标记未知"""
        ref_count = len(self.db.get_labeled_persons_with_features())
        if ref_count == 0:
            QMessageBox.information(
                self, "参考库匹配",
                "参考库为空，请先点击「导入参考库」选择已标记数据集根目录。"
            )
            return
        face_count = len(self.db.get_unassigned_faces_with_features())
        if face_count == 0:
            QMessageBox.information(
                self, "参考库匹配",
                "没有待匹配的未归类人脸。\n请先完成「开始识别」后再执行参考库匹配。"
            )
            return

        threshold = self._ref_thresh_slider.value() / 100.0
        self.logger.info(f"参考库匹配: 阈值={threshold:.2f}，参考库 {ref_count} 人，待匹配 {face_count} 张")
        self._statusbar.showMessage("正在参考库匹配...")
        self._set_actions_enabled(False)

        self._match_progress = QProgressDialog(
            f"正在匹配 {face_count} 张人脸与参考库（阈值 {threshold:.2f}）...",
            None, 0, 1, self
        )
        self._match_progress.setWindowTitle("参考库匹配")
        self._match_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._match_progress.setMinimumDuration(0)

        worker = ReferenceMatchWorker(self.db, cosine_threshold=threshold)
        worker.progress.connect(self._on_ref_match_progress)
        worker.finished_match.connect(self._on_ref_match_done)
        worker.error.connect(lambda msg: self.logger.warning(f"参考库匹配: {msg}"))
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_ref_match_progress(self, current: int, total: int, text: str):
        self._match_progress.setMaximum(max(total, 1))
        self._match_progress.setValue(current)
        self._match_progress.setLabelText(text)

    def _on_ref_match_done(self, result: dict):
        self._match_progress.close()
        self._set_actions_enabled(True)
        matched = result.get("matched", 0)
        unknown = result.get("unknown", 0)
        self._statusbar.showMessage(f"参考库匹配完成: 匹配 {matched} 张，未知 {unknown} 张")
        self._person_panel.refresh()
        if self._current_image_id is not None:
            self._show_image(self._current_image_id)

    # ---- 清空/设置 ----

    def _on_clear_all(self):
        ret = QMessageBox.question(
            self, "确认", "确定要清空所有数据吗？此操作不可撤销。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ret == QMessageBox.StandardButton.Yes:
            self.logger.warning("用户执行清空所有数据操作")
            self.db.clear_all_persons()
            self.db.delete_all_images()
            self._thumb_cache.clear()
            self._image_list.clear()
            self._original_viewer.clear_image()
            self._result_viewer.clear_image()
            self._face_panel.update_faces(None, [])
            self._person_panel.refresh()
            self._statusbar.showMessage("数据已清空")
            self.logger.info("数据清空完成")

    def _on_switch_database(self):
        from ui.pg_connect_dialog import PgConnectDialog

        old_display = self.config.database_display
        dlg = PgConnectDialog(self.config, self)
        if dlg.exec() != PgConnectDialog.DialogCode.Accepted:
            return

        result = dlg.get_result()
        if not result:
            return

        new_display = f"{result['host']}:{result['port']}/{result['database']}"
        if new_display == old_display:
            QMessageBox.information(self, "提示", "数据库连接未改变")
            return

        self.config.set_pg_connection(
            result["host"],
            result["port"],
            result["user"],
            result["password"],
            result["database"],
        )
        self.logger.info(f"用户更改数据库连接: {old_display} -> {new_display}")
        QMessageBox.information(
            self,
            "成功",
            f"数据库连接已切换为:\n{new_display}\n\n"
            f"日志路径: {self.config.log_path}\n\n"
            "请重启程序以应用更改。",
        )

    def _set_actions_enabled(self, enabled: bool):
        self._act_import_images.setEnabled(enabled)
        self._act_import_folder.setEnabled(enabled)
        self._act_detect.setEnabled(enabled)
        self._act_redetect.setEnabled(enabled)
        self._act_cluster.setEnabled(enabled)
        self._act_import_reference.setEnabled(enabled)
        self._act_reference_match.setEnabled(enabled)
        self._act_clear.setEnabled(enabled)
        self._act_switch_db.setEnabled(enabled)
