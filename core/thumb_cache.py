"""缩略图缓存系统 - 文件缓存 + LRU 内存缓存 + 异步加载

核心思路：避免为了显示 56x56 的缩略图而加载数 MB 的完整图片。
检测阶段预先生成缩略图并缓存到磁盘，显示时直接从小文件加载。
"""

import os
from collections import OrderedDict, defaultdict

import cv2
import numpy as np
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QThread, Signal, QMutex, QMutexLocker

from core.logger import get_logger


class ThumbCache:
    """人脸缩略图缓存管理器（文件缓存 + LRU 内存缓存）"""

    THUMB_SIZE = 112          # 存储尺寸（2x 显示尺寸，适配高 DPI）
    MAX_MEMORY_ITEMS = 1000   # 内存 LRU 缓存最大条目数
    JPEG_QUALITY = 85

    def __init__(self, cache_dir: str):
        self._cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._mem: OrderedDict[int, QPixmap] = OrderedDict()
        self._logger = get_logger()

    # ---- 公开接口 ----

    def get_pixmap(self, face_id: int, w: int = 56, h: int = 56) -> QPixmap | None:
        """获取缩略图 QPixmap（优先内存 → 磁盘 → None）"""
        # 1) 内存缓存命中
        if face_id in self._mem:
            self._mem.move_to_end(face_id)
            return self._mem[face_id]

        # 2) 磁盘缓存命中
        path = self._cache_path(face_id)
        if os.path.exists(path):
            pix = QPixmap(path)
            if not pix.isNull():
                scaled = pix.scaled(
                    w, h,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self._put_mem(face_id, scaled)
                return scaled

        return None

    def has_cache(self, face_id: int) -> bool:
        return face_id in self._mem or os.path.exists(self._cache_path(face_id))

    def save_from_crop(self, face_id: int, crop_bgr: np.ndarray):
        """将已裁剪的人脸图像保存到磁盘缓存"""
        if crop_bgr is None or crop_bgr.size == 0:
            return
        h, w = crop_bgr.shape[:2]
        scale = self.THUMB_SIZE / max(h, w, 1)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        path = self._cache_path(face_id)
        ok, buf = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY])
        if ok:
            try:
                buf.tofile(path)
            except OSError as e:
                self._logger.warning(f"写入缩略图缓存失败 face_id={face_id}: {e}")

    def save_from_bytes(self, face_id: int, jpeg_bytes: bytes):
        """直接写入已编码的 JPEG 字节（用于从 worker 线程传递预编码数据）"""
        path = self._cache_path(face_id)
        try:
            with open(path, 'wb') as f:
                f.write(jpeg_bytes)
        except OSError as e:
            self._logger.warning(f"写入缩略图缓存失败 face_id={face_id}: {e}")

    @staticmethod
    def encode_crop(cv_image: np.ndarray, bbox: tuple, padding: float = 0.2) -> bytes | None:
        """在 worker 线程中裁剪+编码人脸缩略图为 JPEG 字节（不需要 face_id）
        
        返回极小的 JPEG 字节（2-5 KB），替代传回完整图像（数 MB）。
        """
        from core.face_engine import FaceEngine
        crop = FaceEngine.crop_face(cv_image, bbox, padding)
        if crop is None or crop.size == 0:
            return None
        h, w = crop.shape[:2]
        scale = ThumbCache.THUMB_SIZE / max(h, w, 1)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, ThumbCache.JPEG_QUALITY])
        return buf.tobytes() if ok else None

    def save_from_image(self, face_id: int, cv_image: np.ndarray, bbox: tuple, padding: float = 0.2):
        """从完整图像中裁剪人脸区域并保存到缓存"""
        from core.face_engine import FaceEngine
        crop = FaceEngine.crop_face(cv_image, bbox, padding)
        self.save_from_crop(face_id, crop)

    def invalidate(self, face_id: int):
        """移除指定人脸的缓存"""
        self._mem.pop(face_id, None)
        path = self._cache_path(face_id)
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass

    def clear(self):
        """清空全部缓存"""
        self._mem.clear()
        try:
            for fname in os.listdir(self._cache_dir):
                if fname.endswith('.jpg'):
                    os.remove(os.path.join(self._cache_dir, fname))
        except OSError:
            pass

    # ---- 内部 ----

    def _cache_path(self, face_id: int) -> str:
        return os.path.join(self._cache_dir, f"{face_id}.jpg")

    def _put_mem(self, face_id: int, pix: QPixmap):
        self._mem[face_id] = pix
        while len(self._mem) > self.MAX_MEMORY_ITEMS:
            self._mem.popitem(last=False)


class ThumbLoaderWorker(QThread):
    """后台缩略图加载线程

    接收一批 (face_id, file_path, bbox) 请求，按 file_path 分组
    以最小化磁盘 I/O，生成缩略图后通过信号通知 UI 更新。
    """

    thumb_ready = Signal(int, QPixmap)   # face_id, pixmap
    batch_done = Signal()

    def __init__(self, cache: ThumbCache, requests: list[tuple], parent=None):
        """
        Args:
            cache: ThumbCache 实例
            requests: [(face_id, file_path, bbox_tuple), ...]
        """
        super().__init__(parent)
        self._cache = cache
        self._requests = requests
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        from core.face_engine import imread_unicode

        # 按文件路径分组（同一张图片的多张人脸只需加载一次）
        by_image: dict[str, list[tuple[int, tuple]]] = defaultdict(list)
        for face_id, file_path, bbox in self._requests:
            # 先检查缓存
            pix = self._cache.get_pixmap(face_id)
            if pix is not None:
                self.thumb_ready.emit(face_id, pix)
            else:
                by_image[file_path].append((face_id, bbox))

        # 逐张图片处理未缓存的缩略图
        for file_path, face_list in by_image.items():
            if self._cancelled:
                break

            img = imread_unicode(file_path)
            if img is None:
                continue

            for face_id, bbox in face_list:
                if self._cancelled:
                    break
                self._cache.save_from_image(face_id, img, bbox)
                pix = self._cache.get_pixmap(face_id)
                if pix is not None:
                    self.thumb_ready.emit(face_id, pix)

            del img  # 尽早释放大图内存

        self.batch_done.emit()
