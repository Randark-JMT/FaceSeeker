"""主窗口 - 组装所有 UI 部件并协调业务逻辑（性能优化版）

优化要点：
1. DetectWorker 使用批量事务写入数据库，减少 commit 次数
2. 检测时预生成缩略图缓存，PersonPanel 显示不再需要读原图
3. 图片列表使用分页加载，避免万级数据一次性渲染
4. 聚类前的预检查移到后台线程，避免阻塞 UI
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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
from core.logger import get_logger, log_opencv_error, log_exception
from core.database import DatabaseManager
from core.face_engine import FaceEngine, FaceData, imread_unicode
from core.face_cluster import FaceCluster
from core.thumb_cache import ThumbCache
from ui.image_viewer import ImageViewer
from ui.face_list_panel import FaceListPanel
from ui.person_panel import PersonPanel


SUPPORTED_FORMATS = "图片文件 (*.jpg *.jpeg *.png *.bmp *.tiff *.webp)"

# 图片列表每次加载的数量
IMAGE_LIST_PAGE_SIZE = 500


# ---- 后台工作线程 ----

class DetectWorker(QThread):
    """后台人脸检测+特征提取线程（批量事务 + 缩略图预生成）"""

    progress = Signal(int, int, str)  # current, total, filename
    finished_all = Signal()
    error = Signal(str)

    def __init__(self, engine: FaceEngine, db: DatabaseManager, file_paths: list[str],
                 base_folder: str = "", thumb_cache: ThumbCache | None = None,
                 num_workers: int | None = None):
        super().__init__()
        self.logger = get_logger()
        self.engine = engine
        self.db = db
        self.file_paths = file_paths
        self.base_folder = base_folder
        self.thumb_cache = thumb_cache
        # CUDA 时并行意义不大（GPU 内部已并行），CPU 时按核心数并行
        if num_workers is None:
            if engine.backend_name == "CUDA":
                self.num_workers = 1
            else:
                self.num_workers = min(os.cpu_count(), 4)
        else:
            self.num_workers = num_workers
        self.logger.info(f"DetectWorker 初始化完成，后端: {engine.backend_name}, 线程数: {self.num_workers}")
        

    def _process_one(self, engine: FaceEngine, fpath: str):
        """在工作线程中处理单张图片（纯计算，不访问数据库）"""
        image = imread_unicode(fpath)
        if image is None:
            return None

        h, w = image.shape[:2]
        faces = engine.detect(image)
        engine.extract_features(image, faces)

        # 计算相对路径
        relative_path = ""
        if self.base_folder:
            try:
                relative_path = os.path.relpath(os.path.dirname(fpath), self.base_folder)
                if relative_path == ".":
                    relative_path = ""
            except ValueError:
                relative_path = ""

        return fpath, os.path.basename(fpath), w, h, faces, relative_path, image

    def run(self):
        total = len(self.file_paths)

        # 预过滤已存在的图片
        pending_paths = [p for p in self.file_paths if not self.db.image_exists(p)]
        skipped = total - len(pending_paths)
        if skipped > 0:
            self.logger.info(f"跳过 {skipped} 张已存在的图片")

        # 多线程引擎
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

        counter = [0]

        def worker(idx: int, fpath: str):
            engine = _get_thread_engine()
            return self._process_one(engine, fpath)

        # ★ 使用批量事务提交，大幅减少 I/O
        BATCH_COMMIT_SIZE = 50
        uncommitted = 0
        self.db.begin()

        try:
            with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                futures = {
                    pool.submit(worker, i, fpath): fpath
                    for i, fpath in enumerate(pending_paths)
                }

                for future in as_completed(futures):
                    fpath = futures[future]
                    counter[0] += 1
                    self.progress.emit(skipped + counter[0], total, os.path.basename(fpath))

                    try:
                        result = future.result()
                        if result is None:
                            continue

                        fpath, filename, w, h, faces, relative_path, image = result

                        # 写入图片记录
                        image_id = self.db.add_image(fpath, filename, w, h, len(faces), relative_path)

                        # 批量写入人脸记录 + 预生成缩略图
                        for face in faces:
                            face_id = self.db.add_face(
                                image_id, face.bbox, face.landmarks,
                                face.score, face.feature,
                            )
                            # ★ 预生成缩略图缓存
                            if self.thumb_cache and image is not None:
                                self.thumb_cache.save_from_image(face_id, image, face.bbox)

                        del image  # 尽早释放大图内存

                        uncommitted += 1
                        if uncommitted >= BATCH_COMMIT_SIZE:
                            self.db.commit()
                            self.db.begin()
                            uncommitted = 0

                    except Exception as e:
                        log_opencv_error("DetectWorker._process_one", e, suppress=True)
                        self.error.emit(f"{os.path.basename(fpath)}: {e}")

            # 提交剩余数据
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

        self.finished_all.emit()


class ClusterWorker(QThread):
    """后台聚类线程"""

    progress = Signal(int, int, str)  # current, total, stage_text
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

        # ★ 初始化缩略图缓存
        thumb_dir = os.path.join(config.data_dir, "thumb_cache")
        self._thumb_cache = ThumbCache(thumb_dir)

        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.resize(1280, 800)

        self._build_toolbar()
        self._build_ui()
        self._build_statusbar()

        # ★ 延迟加载：窗口显示后再加载数据，避免启动卡顿
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
        self._act_detect.triggered.connect(self._on_detect_all)
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
        self._thresh_slider.setRange(20, 60)
        self._thresh_slider.setValue(36)
        self._thresh_slider.setFixedWidth(120)
        self._thresh_slider.setToolTip("调整人脸聚类的余弦相似度阈值\n值越高→分类越严格，人物组越多\n值越低→分类越宽松，更容易合并")
        tb.addWidget(self._thresh_slider)
        self._thresh_value_label = QLabel("0.36")
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
        il_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 4px;")
        il_layout.addWidget(il_label)
        self._image_list = QListWidget()
        self._image_list.currentItemChanged.connect(self._on_image_selected)
        il_layout.addWidget(self._image_list)
        bottom_splitter.addWidget(image_list_container)

        # 人脸列表
        self._face_panel = FaceListPanel(thumb_cache=self._thumb_cache)
        bottom_splitter.addWidget(self._face_panel)

        # 人物归类（传入缩略图缓存）
        self._person_panel = PersonPanel(self.db, thumb_cache=self._thumb_cache)
        self._person_panel.navigate_to_image.connect(self._navigate_to_image)
        bottom_splitter.addWidget(self._person_panel)

        bottom_splitter.setStretchFactor(0, 1)
        bottom_splitter.setStretchFactor(1, 1)
        bottom_splitter.setStretchFactor(2, 2)  # ★ 人物面板更宽，减少挤压

        # 主分割器（上下）
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
        """从数据库加载图片列表（分批加载避免卡顿）"""
        self._image_list.clear()

        # ★ 分批加载：每批 PAGE_SIZE 条，避免一次性创建万级 QListWidgetItem
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
                item = QListWidgetItem(f"{display_name}  [{row['face_count']} 人脸]")
                item.setData(Qt.ItemDataRole.UserRole, row["id"])
                self._image_list.addItem(item)
            total_loaded += len(rows)
            offset += IMAGE_LIST_PAGE_SIZE

            # 每批之间让 UI 呼吸一下
            if total_loaded % 2000 == 0:
                QApplication.processEvents()

        # 加载人物归类数据
        self._person_panel.refresh()

        image_count = self.db.get_image_count()
        if image_count > 0:
            self._statusbar.showMessage(f"已加载 {image_count} 张图片")

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

        # 加载人脸数据
        face_rows = self.db.get_faces_by_image(image_id)
        faces_data = [dict(row) for row in face_rows]

        # 生成标注图
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

        # 更新人脸面板
        self._face_panel.update_faces(cv_img, faces_data)

    def _navigate_to_image(self, image_id: int):
        """双击人脸缩略图后跳转到对应图片"""
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
            self._run_detect(paths)

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
            self.logger.warning(f"文件夹 {folder} 中没有找到支持的图片格式")
            QMessageBox.information(self, "提示", "该文件夹及其子文件夹下没有找到支持的图片格式。")
            return
        self.logger.info(f"扫描到 {len(paths)} 张图片")
        self._run_detect(paths, folder)

    def _on_detect_all(self):
        """对数据库中所有未识别的图片重新识别"""
        images = self.db.get_all_images()
        if not images:
            QMessageBox.information(self, "提示", "请先导入图片。")
            return
        self.logger.info(f"开始重新识别 {len(images)} 张图片")
        paths = [row["file_path"] for row in images]
        # ★ 批量清空（替代逐条删除）
        self.db.delete_all_images()
        # 清除缩略图缓存
        self._thumb_cache.clear()
        self._run_detect(paths)

    def _on_cluster(self):
        # ★ 使用轻量计数查询替代加载全部数据
        face_count = self.db.get_face_count()
        if face_count == 0:
            QMessageBox.information(self, "提示", "没有可用的人脸特征数据，请先导入并识别图片。")
            return

        threshold = self._thresh_slider.value() / 100.0
        self.logger.info(f"开始人脸聚类，阈值={threshold:.2f}，共 {face_count} 张人脸")

        self._statusbar.showMessage("正在进行人脸归类...")
        self._set_actions_enabled(False)

        self._cluster_progress = QProgressDialog(
            f"正在归类 {face_count} 张人脸（阈值 {threshold:.2f}）...", None, 0, 100, self
        )
        self._cluster_progress.setWindowTitle("人脸归类")
        self._cluster_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._cluster_progress.setMinimumDuration(0)
        self._cluster_progress.setValue(0)

        worker = ClusterWorker(self.cluster_engine, threshold, incremental=True)
        worker.progress.connect(self._on_cluster_progress)
        worker.finished_cluster.connect(self._on_cluster_done)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_cluster_progress(self, current: int, total: int, text: str):
        self._cluster_progress.setMaximum(max(total, 1))
        self._cluster_progress.setValue(current)
        self._cluster_progress.setLabelText(text)

    def _on_clear_all(self):
        ret = QMessageBox.question(
            self, "确认", "确定要清空所有数据吗？此操作不可撤销。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ret == QMessageBox.StandardButton.Yes:
            self.logger.warning("用户执行清空所有数据操作")
            self.db.clear_all_persons()
            # ★ 批量清空替代逐条删除
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
        """设置数据目录"""
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
            self.logger.info(f"数据目录已更新，新的数据库路径: {self.config.database_path}")
            QMessageBox.information(
                self, "成功",
                f"数据目录已更改为:\n{self.config.data_dir}\n\n"
                f"数据库路径: {self.config.database_path}\n"
                f"日志路径: {self.config.log_path}\n\n"
                "请重启程序以应用更改。"
            )
        except Exception as e:
            self.logger.error(f"设置数据目录失败: {e}", exc_info=True)
            QMessageBox.critical(
                self, "错误",
                f"设置数据目录失败:\n{str(e)}"
            )

    # ---- 后台检测 ----

    def _run_detect(self, paths: list[str], base_folder: str = ""):
        self._statusbar.showMessage("正在检测人脸...")
        self._set_actions_enabled(False)

        self._progress = QProgressDialog("正在处理图片...", "取消", 0, len(paths), self)
        self._progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress.setMinimumDuration(0)

        # ★ 传入缩略图缓存，检测时自动预生成
        worker = DetectWorker(self.engine, self.db, paths, base_folder,
                              thumb_cache=self._thumb_cache)
        worker.progress.connect(self._on_detect_progress)
        worker.finished_all.connect(self._on_detect_done)
        worker.error.connect(lambda msg: self._statusbar.showMessage(f"错误: {msg}"))
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_detect_progress(self, current: int, total: int, filename: str):
        self._progress.setLabelText(f"处理 [{current}/{total}]: {filename}")
        self._progress.setValue(current)

    def _on_detect_done(self):
        self._progress.close()
        self._load_image_list()
        self._set_actions_enabled(True)
        total_images = self._image_list.count()
        self._statusbar.showMessage(f"检测完成，共 {total_images} 张图片")

        if total_images > 0:
            self._image_list.setCurrentRow(0)

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

    def _set_actions_enabled(self, enabled: bool):
        self._act_import_images.setEnabled(enabled)
        self._act_import_folder.setEnabled(enabled)
        self._act_detect.setEnabled(enabled)
        self._act_cluster.setEnabled(enabled)
        self._act_clear.setEnabled(enabled)
