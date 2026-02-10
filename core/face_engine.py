"""人脸检测与识别引擎，基于 OpenCV YuNet + SFace"""

import os
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from core.logger import get_logger, log_opencv_error


@dataclass
class FaceData:
    """单个人脸的检测数据"""

    bbox: tuple  # (x, y, w, h)
    landmarks: list  # 5 个关键点 [(x,y), ...]
    score: float
    feature: Optional[np.ndarray] = field(default=None, repr=False)

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
        # 验证图像有效性：不为 None，维度正确，且有实际内容
        if img is None or img.size == 0 or len(img.shape) < 2:
            return None
        # 归一化图像格式，确保 YuNet / SFace 能正确处理
        img = _normalize_image(img)
        return img
    except Exception as e:
        log_opencv_error("imread_unicode", e, suppress=True)
        return None


def _normalize_image(img: np.ndarray) -> np.ndarray | None:
    """将图像归一化为 3 通道 uint8 BGR 连续数组。
    
    处理灰度、RGBA、16-bit 等非标准格式，返回 None 表示无法转换。
    """
    if img is None or img.size == 0:
        return None

    # 转换为 uint8（处理 16-bit / float 图像）
    if img.dtype != np.uint8:
        if img.dtype in (np.float32, np.float64):
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
        elif img.dtype == np.uint16:
            img = (img >> 8).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    # 转换通道数
    if len(img.shape) == 2:
        # 灰度 → BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3:
        channels = img.shape[2]
        if channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif channels == 4:
            # RGBA / BGRA → BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif channels != 3:
            return None  # 不支持的通道数
    else:
        return None

    # 确保内存连续（某些 OpenCV DNN 操作要求 contiguous）
    if not img.flags['C_CONTIGUOUS']:
        img = np.ascontiguousarray(img)

    return img


def detect_backends() -> tuple[int, int, str]:
    """检测可用的 DNN 后端，优先 CUDA > CPU。返回 (backend_id, target_id, name)"""
    logger = get_logger()
    try:
        # 尝试创建一个小的 CUDA 矩阵来验证 CUDA 是否真正可用
        backends = cv2.dnn.getAvailableBackends()
        cuda_pair = (cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA)
        if cuda_pair in backends:
            logger.info("检测到 CUDA 后端可用")
            return cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA, "CUDA"
    except Exception as e:
        log_opencv_error("detect_backends", e, suppress=True)
    
    logger.info("使用 CPU 后端")
    return cv2.dnn.DNN_BACKEND_DEFAULT, cv2.dnn.DNN_TARGET_CPU, "CPU"


# 关键点颜色 (BGR)
LANDMARK_COLORS = [
    (255, 0, 0),    # 右眼
    (0, 0, 255),    # 左眼
    (0, 255, 0),    # 鼻子
    (255, 0, 255),  # 右嘴角
    (0, 255, 255),  # 左嘴角
]


class FaceEngine:
    """人脸检测 + 识别引擎"""

    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    # 最小人脸尺寸（宽或高小于此值的检测结果将被丢弃，太小的人脸特征不可靠）
    MIN_FACE_SIZE = 50

    def __init__(
        self,
        detection_model: str,
        recognition_model: str,
        score_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        backend_id: int | None = None,
        target_id: int | None = None,
    ):
        self.logger = get_logger()
        
        if not os.path.exists(detection_model):
            raise FileNotFoundError(
                f"找不到检测模型: {detection_model}\n"
                "请从 https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet 下载"
            )
        if not os.path.exists(recognition_model):
            raise FileNotFoundError(
                f"找不到识别模型: {recognition_model}\n"
                "请从 https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface 下载"
            )

        # 保存参数用于 clone()
        self._detection_model = detection_model
        self._recognition_model = recognition_model
        self._score_threshold = score_threshold
        self._nms_threshold = nms_threshold
        self._top_k = top_k

        if backend_id is None:
            backend_id, target_id, self.backend_name = detect_backends()
        else:
            self.backend_name = "CUDA" if target_id == cv2.dnn.DNN_TARGET_CUDA else "CPU"

        self._backend_id = backend_id
        self._target_id = target_id
        
        self.logger.info(f"初始化人脸检测器: {detection_model}")
        self.logger.info(f"初始化人脸识别器: {recognition_model}")

        self.detector = cv2.FaceDetectorYN.create(
            detection_model, "", (320, 320),
            score_threshold, nms_threshold, top_k,
            backend_id, target_id,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(
            recognition_model, "", backend_id, target_id,
        )

    def clone(self) -> "FaceEngine":
        """创建一个独立的引擎实例（用于多线程，每个线程需要自己的 detector 实例）"""
        return FaceEngine(
            self._detection_model, self._recognition_model,
            self._score_threshold, self._nms_threshold, self._top_k,
            self._backend_id, self._target_id,
        )

    def detect(self, image: np.ndarray, min_face_size: int | None = None) -> list[FaceData]:
        """检测图像中所有人脸

        Args:
            image: 输入图像 (BGR)
            min_face_size: 最小人脸尺寸，低于此值的检测结果会被过滤。
                           None 时使用类默认值 MIN_FACE_SIZE。
        """
        # 验证输入图像的有效性
        if image is None or image.size == 0 or len(image.shape) < 2:
            self.logger.warning("detect: 输入图像无效")
            return []
        
        try:
            h, w = image.shape[:2]
            if h <= 0 or w <= 0:
                self.logger.warning(f"detect: 图像尺寸无效 ({w}x{h})")
                return []

            # 防御性检查：确保图像是 3 通道 uint8（避免 OpenCV 断言失败）
            if image.dtype != np.uint8 or len(image.shape) != 3 or image.shape[2] != 3:
                image = _normalize_image(image)
                if image is None:
                    self.logger.warning("detect: 图像格式无法转换为 BGR uint8")
                    return []

            if not image.flags['C_CONTIGUOUS']:
                image = np.ascontiguousarray(image)

            self.detector.setInputSize((w, h))
            _, raw_faces = self.detector.detect(image)

            if raw_faces is None or len(raw_faces) == 0:
                return []

            min_sz = min_face_size if min_face_size is not None else self.MIN_FACE_SIZE

            results = []
            for face_row in raw_faces:
                try:
                    # 验证数值有效性（过滤 inf/nan）
                    if not all(np.isfinite(face_row[:4])):
                        continue
                    
                    bbox = tuple(map(int, face_row[:4]))
                    fw, fh = bbox[2], bbox[3]
                    # 过滤过小的人脸（特征不可靠）
                    if fw < min_sz or fh < min_sz:
                        continue

                    # 验证关键点数值有效性
                    landmark_coords = face_row[4:14]
                    if not all(np.isfinite(landmark_coords)):
                        continue
                    
                    landmarks = [
                        (int(face_row[4 + j * 2]), int(face_row[5 + j * 2]))
                        for j in range(5)
                    ]
                    score = float(face_row[14])
                    results.append(FaceData(bbox=bbox, landmarks=landmarks, score=score))
                except (ValueError, OverflowError) as e:
                    # 跳过无效的检测结果
                    self.logger.debug(f"跳过无效检测结果: {e}")
                    continue
            return results
        except Exception as e:
            log_opencv_error("FaceEngine.detect", e, suppress=True)
            return []

    def extract_feature(self, image: np.ndarray, face: FaceData) -> np.ndarray:
        """提取单张人脸的特征向量（L2 归一化）"""
        # 验证输入图像的有效性
        if image is None or image.size == 0 or len(image.shape) < 2:
            self.logger.warning("extract_feature: 输入图像无效")
            face.feature = None
            return np.zeros(128, dtype=np.float32)
        
        try:
            aligned = self.recognizer.alignCrop(image, face.to_detect_array())
            # 验证对齐后的图像是否有效
            if aligned is None or aligned.size == 0:
                self.logger.warning("extract_feature: 人脸对齐失败")
                face.feature = None
                return np.zeros(128, dtype=np.float32)
            
            feature = self.recognizer.feature(aligned)
            # L2 归一化，确保后续余弦相似度计算的数值稳定性
            norm = np.linalg.norm(feature)
            if norm > 0:
                feature = feature / norm
            face.feature = feature
            return feature
        except Exception as e:
            log_opencv_error("FaceEngine.extract_feature", e, suppress=True)
            # 返回一个零向量
            face.feature = None
            return np.zeros(128, dtype=np.float32)

    def extract_features(self, image: np.ndarray, faces: list[FaceData]) -> list[FaceData]:
        """批量提取人脸特征"""
        for face in faces:
            self.extract_feature(image, face)
        return faces

    def compare(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """比较两个特征向量的余弦相似度"""
        return float(
            self.recognizer.match(feat1, feat2, cv2.FaceRecognizerSF_FR_COSINE)
        )

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
