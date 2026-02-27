"""人脸检测与识别引擎，基于 facenet-pytorch（MTCNN + InceptionResnetV1）"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional

from facenet_pytorch import MTCNN, InceptionResnetV1

from core.logger import get_logger, log_opencv_error


# FaceNet 特征向量维度（InceptionResnetV1 输出）
FEATURE_DIM = 512


@dataclass
class FaceData:
    """单个人脸的检测数据"""

    bbox: tuple  # (x, y, w, h)
    landmarks: list  # 5 个关键点 [(x,y), ...]
    score: float
    feature: Optional[np.ndarray] = field(default=None, repr=False)
    blur_score: float = 0.0  # 清晰度分数（Laplacian 方差），越高越清晰

    def to_detect_array(self) -> np.ndarray:
        flat_landmarks = [coord for pt in self.landmarks for coord in pt]
        return np.array(
            [*self.bbox, *flat_landmarks, self.score], dtype=np.float32
        )


def imread_unicode(filepath: str) -> np.ndarray | None:
    """读取 Unicode 路径的图像（解决 Windows 上 cv2.imread 不支持非 ASCII 路径的问题）

    返回的图像保证为 3 通道 uint8 BGR 连续数组，或 None。
    """
    try:
        buf = np.fromfile(filepath, dtype=np.uint8)
        if buf.size == 0:
            return None
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None or img.size == 0 or len(img.shape) < 2:
            return None
        img = _normalize_image(img)
        return img
    except Exception as e:
        log_opencv_error("imread_unicode", e, suppress=True)
        return None


def _normalize_image(img: np.ndarray) -> np.ndarray | None:
    """将图像归一化为 3 通道 uint8 BGR 连续数组。"""
    if img is None or img.size == 0:
        return None

    if img.dtype != np.uint8:
        if img.dtype in (np.float32, np.float64):
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
        elif img.dtype == np.uint16:
            img = (img >> 8).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3:
        channels = img.shape[2]
        if channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif channels == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif channels != 3:
            return None
    else:
        return None

    if not img.flags["C_CONTIGUOUS"]:
        img = np.ascontiguousarray(img)
    return img


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """BGR -> RGB，返回连续数组"""
    out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if not out.flags["C_CONTIGUOUS"]:
        out = np.ascontiguousarray(out)
    return out


def _get_device(device: Optional[torch.device] = None) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 关键点颜色 (BGR)
LANDMARK_COLORS = [
    (255, 0, 0),    # 右眼
    (0, 0, 255),    # 左眼
    (0, 255, 0),    # 鼻子
    (255, 0, 255),  # 右嘴角
    (0, 255, 255),  # 左嘴角
]


def _prewhiten(x: np.ndarray) -> np.ndarray:
    """FaceNet 输入预处理：归一化到约 [-1, 1]（uint8 [0,255] -> (x-127.5)/128）"""
    return (x.astype(np.float32) - 127.5) / 128.0


class FaceEngine:
    """人脸检测 + 识别引擎（MTCNN + InceptionResnetV1）"""

    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    MIN_FACE_SIZE = 30

    def __init__(
        self,
        detection_input_max_side: int = 0,
        device: Optional[torch.device] = None,
        mtcnn_image_size: int = 160,
        mtcnn_margin: int = 0,
        mtcnn_thresholds: tuple = (0.6, 0.7, 0.7),
        pretrained: str = "vggface2",
    ):
        """
        Args:
            detection_input_max_side: 检测时输入图像最长边上限，0 表示不限制。
            device: PyTorch 设备，None 时自动选择 CUDA（若可用）或 CPU。
            mtcnn_image_size: MTCNN 输出人脸尺寸，需与 InceptionResnet 一致，默认 160。
            mtcnn_margin: 人脸框外扩像素。
            mtcnn_thresholds: MTCNN 三阶段阈值。
            pretrained: InceptionResnet 预训练集，'vggface2' 或 'casia-webface'。
        """
        self.logger = get_logger()
        self._device = _get_device(device)
        self.backend_name = "CUDA" if self._device.type == "cuda" else "CPU"
        self.logger.info(f"FaceEngine 初始化成功 [后端: {self.backend_name}]")
        self._detection_input_max_side = max(0, detection_input_max_side)
        self._mtcnn_image_size = mtcnn_image_size
        self._mtcnn_margin = mtcnn_margin
        self._mtcnn_thresholds = mtcnn_thresholds
        self._pretrained = pretrained

        self.logger.info("初始化 MTCNN 人脸检测器...")
        self.mtcnn = MTCNN(
            image_size=mtcnn_image_size,
            margin=mtcnn_margin,
            thresholds=mtcnn_thresholds,
            keep_all=True,
            device=self._device,
        )

        self.logger.info("初始化 InceptionResnetV1 人脸识别模型...")
        self.resnet = InceptionResnetV1(pretrained=pretrained).eval().to(self._device)

    def clone(self) -> "FaceEngine":
        """创建独立的引擎实例（用于多线程）"""
        return FaceEngine(
            detection_input_max_side=self._detection_input_max_side,
            device=self._device,
            mtcnn_image_size=self._mtcnn_image_size,
            mtcnn_margin=self._mtcnn_margin,
            mtcnn_thresholds=self._mtcnn_thresholds,
            pretrained=self._pretrained,
        )

    def _prepare_detect_image(self, image: np.ndarray) -> tuple[Image.Image, float]:
        """BGR numpy -> RGB PIL，可选缩放，返回 (pil_image, scale)."""
        h, w = image.shape[:2]
        scale = 1.0
        if self._detection_input_max_side > 0 and max(w, h) > self._detection_input_max_side:
            scale = self._detection_input_max_side / max(w, h)
            nw, nh = int(round(w * scale)), int(round(h * scale))
            if nw < 1:
                nw = 1
            if nh < 1:
                nh = 1
            image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb = _bgr_to_rgb(image)
        pil = Image.fromarray(rgb)
        return pil, scale

    def detect(self, image: np.ndarray, min_face_size: int | None = None) -> list[FaceData]:
        """检测图像中所有人脸。

        Args:
            image: 输入图像 (BGR, numpy).
            min_face_size: 最小人脸尺寸，低于此值的检测结果会被过滤；None 时使用 MIN_FACE_SIZE。
        """
        if image is None or image.size == 0 or len(image.shape) < 2:
            self.logger.warning("detect: 输入图像无效")
            return []

        try:
            if image.dtype != np.uint8 or len(image.shape) != 3 or image.shape[2] != 3:
                image = _normalize_image(image)
                if image is None:
                    self.logger.warning("detect: 图像格式无法转换为 BGR uint8")
                    return []

            if not image.flags["C_CONTIGUOUS"]:
                image = np.ascontiguousarray(image)

            pil_img, scale = self._prepare_detect_image(image)
            boxes, probs, landmarks = self.mtcnn.detect(pil_img, landmarks=True)

            if boxes is None or len(boxes) == 0:
                return []

            min_sz = min_face_size if min_face_size is not None else self.MIN_FACE_SIZE
            inv_scale = 1.0 / scale
            results = []

            for i in range(len(boxes)):
                box = boxes[i]
                # MTCNN 可能返回 Tensor，需转为 numpy 再使用 np.isfinite
                box = np.asarray(box, dtype=np.float64)
                prob = probs[i] if probs is not None else 1.0
                prob = float(prob) if prob is not None else 1.0
                if not np.isfinite(prob):
                    continue
                if not np.all(np.isfinite(box)):
                    continue

                x1, y1, x2, y2 = box
                if scale != 1.0:
                    x1, y1, x2, y2 = x1 * inv_scale, y1 * inv_scale, x2 * inv_scale, y2 * inv_scale
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)
                if w < min_sz or h < min_sz:
                    continue

                bbox = (x, y, w, h)
                if landmarks is not None and i < len(landmarks):
                    lm = landmarks[i]
                    if scale != 1.0:
                        lm = lm * inv_scale
                    landlist = [(int(lm[j, 0]), int(lm[j, 1])) for j in range(5)]
                else:
                    landlist = [(0, 0)] * 5

                results.append(FaceData(bbox=bbox, landmarks=landlist, score=float(prob)))
            return results

        except Exception as e:
            self.logger.warning("FaceEngine.detect: %s", e, exc_info=True)
            return []

    def _crop_face_for_resnet(self, image: np.ndarray, face: FaceData, margin: float = 0.2) -> torch.Tensor | None:
        """根据 bbox 裁剪人脸并预处理为 ResNet 输入：160x160，prewhiten，NCHW tensor。"""
        x, y, w, h = face.bbox
        img_h, img_w = image.shape[:2]
        pad_w = int(w * margin)
        pad_h = int(h * margin)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img_w, x + w + pad_w)
        y2 = min(img_h, y + h + pad_h)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        # BGR -> RGB, resize 160x160
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_resized = cv2.resize(crop_rgb, (self._mtcnn_image_size, self._mtcnn_image_size), interpolation=cv2.INTER_LINEAR)
        # prewhiten, HWC -> CHW, float32 tensor
        normalized = _prewhiten(crop_resized)
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float().to(self._device)
        return tensor

    def extract_feature(self, image: np.ndarray, face: FaceData) -> np.ndarray:
        """提取单张人脸的特征向量（512 维，L2 归一化）。"""
        if image is None or image.size == 0 or len(image.shape) < 2:
            self.logger.warning("extract_feature: 输入图像无效")
            face.feature = None
            return np.zeros(FEATURE_DIM, dtype=np.float32)

        try:
            tensor = self._crop_face_for_resnet(image, face)
            if tensor is None:
                self.logger.warning("extract_feature: 人脸裁剪失败")
                face.feature = None
                return np.zeros(FEATURE_DIM, dtype=np.float32)

            with torch.no_grad():
                embedding = self.resnet(tensor)
            feature = embedding.cpu().numpy().flatten().astype(np.float32)
            norm = np.linalg.norm(feature)
            if norm > 0:
                feature = feature / norm
            face.feature = feature
            return feature
        except Exception as e:
            self.logger.warning("FaceEngine.extract_feature: %s", e, exc_info=True)
            face.feature = None
            return np.zeros(FEATURE_DIM, dtype=np.float32)

    def extract_features(self, image: np.ndarray, faces: list[FaceData]) -> list[FaceData]:
        """批量提取人脸特征"""
        for face in faces:
            self.extract_feature(image, face)
        return faces

    def compare(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """比较两个特征向量的余弦相似度（特征应为 L2 归一化，则余弦相似度 = 内积）。"""
        if feat1.size == 0 or feat2.size == 0:
            return 0.0
        a, b = feat1.flatten(), feat2.flatten()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def visualize(
        self,
        image: np.ndarray,
        faces: list[FaceData],
        person_ids: Optional[list] = None,
        thickness: int = 2,
    ) -> np.ndarray:
        """在图像上绘制人脸框、关键点、人物ID"""
        result = image.copy()
        for idx, face in enumerate(faces):
            x, y, w, h = face.bbox
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), thickness)

            if person_ids and idx < len(person_ids) and person_ids[idx] is not None:
                label = f"P{person_ids[idx]} ({face.score:.2f})"
            else:
                label = f"#{idx} ({face.score:.2f})"
            cv2.putText(result, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            for j, (lx, ly) in enumerate(face.landmarks):
                cv2.circle(result, (lx, ly), 3, LANDMARK_COLORS[j], thickness)

        cv2.putText(result, f"Faces: {len(faces)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return result

    @staticmethod
    def compute_blur_score(crop: np.ndarray) -> float:
        """计算人脸清晰度质量分数（多指标融合），0-100。"""
        if crop is None or crop.size == 0:
            return 0.0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        h, w = gray.shape
        if h < 10 or w < 10:
            return 0.0

        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = float(np.sqrt(gx * gx + gy * gy).mean())

        f = np.fft.fft2(gray.astype(np.float64))
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        cy, cx = h // 2, w // 2
        radius = min(h, w) / 3.0
        Y, X = np.ogrid[:h, :w]
        dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
        radius_sq = radius * radius
        total_energy = float(magnitude.sum()) + 1e-10
        high_freq_energy = float(magnitude[dist_sq > radius_sq].sum())
        hf_ratio = high_freq_energy / total_energy

        def _sigmoid(x: float, center: float, scale: float) -> float:
            z = -(x - center) / max(scale, 1e-8)
            if z > 30:
                return 0.0
            if z < -30:
                return 1.0
            return 1.0 / (1.0 + np.exp(z))

        lap_score = _sigmoid(laplacian_var, 80, 40)
        ten_score = _sigmoid(tenengrad, 25, 12)
        hf_score = _sigmoid(hf_ratio, 0.35, 0.08)
        quality = (lap_score * 0.25 + ten_score * 0.35 + hf_score * 0.40) * 100
        return round(quality, 1)

    @staticmethod
    def crop_face(image: np.ndarray, bbox: tuple, padding: float = 0.2) -> np.ndarray:
        """裁剪人脸区域（带 padding）"""
        x, y, w, h = bbox
        img_h, img_w = image.shape[:2]
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img_w, x + w + pad_w)
        y2 = min(img_h, y + h + pad_h)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return image[y:y + h, x:x + w]
        return crop
