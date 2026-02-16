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
                 num_workers: int | None = None):
        """
        Args:
            image_items: [(image_id, file_path), ...] 待检测的图片列表
        """
        super().__init__()
        self.logger = get_logger()
        self.engine = engine
        self.db = db
        self.image_items = image_items
        self.thumb_cache = thumb_cache
        if num_workers is None:
            if engine.backend_name == "CUDA":
                self.num_workers = 1
            else:
                self.num_workers = min(os.cpu_count() or 4, 4)
        else:
            self.num_workers = num_workers
        self.logger.info(f"DetectWorker: {len(image_items)} 张待检测，线程数: {self.num_workers}")

    def _process_one(self, engine: FaceEngine, fpath: str):
        """在 worker 线程中处理单张图片，返回轻量结果（不含完整图像）"""
        image = imread_unicode(fpath)
        if image is None:
            return None

        h, w = image.shape[:2]
        faces = engine.detect(image)
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
        self.cluster_engine = FaceCluster(db, engine.recognizer)
        self.logger = get_logger()

        self._current_image_id: int | None = None
        self._worker: QThread | None = None

        # 缩略图缓存
        thumb_dir = os.path.join(config.data_dir, "thumb_cache")
        self._thumb_cache = ThumbCache(thumb_dir)

        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.resize(1280, 800)

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

        self._act_cluster = QAction("人脸归类", self)
        self._act_cluster.triggered.connect(self._on_cluster)
        tb.addAction(self._act_cluster)

        tb.addSeparator()

        self._act_clear = QAction("清空数据", self)
        self._act_clear.triggered.connect(self._on_clear_all)
        tb.addAction(self._act_clear)

        tb.addSeparator()

        self._act_set_cache = QAction("设置数据目录", self)
        self._act_set_cache.triggered.connect(self._on_set_cache_dir)
        tb.addAction(self._act_set_cache)

        tb.addSeparator()

        # 聚类阈值滑块
        thresh_label = QLabel("聚类阈值:")
        thresh_label.setStyleSheet("padding: 0 4px;")
        tb.addWidget(thresh_label)
        self._thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self._thresh_slider.setRange(20, 80)
        self._thresh_slider.setValue(60)
        self._thresh_slider.setFixedWidth(120)
        self._thresh_slider.setToolTip("调整人脸聚类的余弦相似度阈值\n值越高→分类越严格，人物组越多\n值越低→分类越宽松，更容易合并")
        tb.addWidget(self._thresh_slider)
        self._thresh_value_label = QLabel("0.60")
        self._thresh_value_label.setFixedWidth(36)
        self._thresh_value_label.setStyleSheet("padding: 0 4px;")
        tb.addWidget(self._thresh_value_label)
        self._thresh_slider.valueChanged.connect(
            lambda v: self._thresh_value_label.setText(f"{v / 100:.2f}")
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

        bottom_splitter.setStretchFactor(0, 1)
        bottom_splitter.setStretchFactor(1, 1)
        bottom_splitter.setStretchFactor(2, 2)

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

        worker = DetectWorker(
            self.engine, self.db, items,
            thumb_cache=self._thumb_cache,
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

    def _on_set_cache_dir(self):
        from ui.data_dir_dialog import DataDirDialog

        dlg = DataDirDialog(self.config, self)
        if dlg.exec() != DataDirDialog.DialogCode.Accepted:
            return

        new_dir = dlg.get_selected_dir()
        effective_new = new_dir if new_dir else self.config.default_data_dir
        current_dir = self.config.data_dir

        if os.path.normpath(effective_new) == os.path.normpath(current_dir):
            QMessageBox.information(self, "提示", "数据目录未改变")
            return

        self.logger.info(f"用户更改数据目录: {current_dir} -> {effective_new}")
        try:
            self.config.set_data_dir(new_dir)
            QMessageBox.information(
                self, "成功",
                f"数据目录已更改为:\n{self.config.data_dir}\n\n"
                f"数据库路径: {self.config.database_path}\n"
                f"日志路径: {self.config.log_path}\n\n"
                "请重启程序以应用更改。"
            )
        except Exception as e:
            self.logger.error(f"设置数据目录失败: {e}", exc_info=True)
            QMessageBox.critical(self, "错误", f"设置数据目录失败:\n{str(e)}")

    def _set_actions_enabled(self, enabled: bool):
        self._act_import_images.setEnabled(enabled)
        self._act_import_folder.setEnabled(enabled)
        self._act_detect.setEnabled(enabled)
        self._act_cluster.setEnabled(enabled)
        self._act_clear.setEnabled(enabled)
