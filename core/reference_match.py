"""参考库匹配：以 labeled_persons（已知特征）为标准，与人脸相似度比对，按阈值匹配分组，未匹配放入未知"""

import numpy as np
from PySide6.QtCore import QThread, Signal

from core.database import DatabaseManager
from core.logger import get_logger

# 人脸分批大小，控制单次加载数量以降低内存峰值（百万级时约 1 万/批）
FACE_CHUNK_SIZE = 10000


def search_reference_top_k(db: DatabaseManager, face_id: int, top_k: int = 20) -> list[tuple[dict, float]]:
    """
    对单张人脸在参考库中检索相似度最高的 top_k 条，按相似度从高到低返回。
    返回: [(ref_row_dict, similarity), ...]，ref_row_dict 为 labeled_persons 行（含 person_id, folder_path, feature 等）。
    若该人脸无特征或参考库为空，返回空列表。
    """
    face_row = db.get_face(face_id)
    if not face_row or not face_row.get("feature"):
        return []
    refs = db.get_labeled_persons_with_features()
    if not refs:
        return []

    face_vec = DatabaseManager.feature_from_blob(face_row["feature"]).flatten().astype(np.float32)
    fnorm = np.linalg.norm(face_vec)
    if fnorm < 1e-10:
        return []
    face_vec = (face_vec / fnorm).reshape(1, -1)

    ref_matrix = np.vstack([
        DatabaseManager.feature_from_blob(r["feature"]).flatten().astype(np.float32)
        for r in refs
    ])
    norms = np.linalg.norm(ref_matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    ref_matrix = ref_matrix / norms

    sim = (ref_matrix @ face_vec.T).flatten()
    n = min(top_k, len(sim))
    if n == 0:
        return []
    top_indices = np.argsort(sim)[::-1][:n]
    return [(refs[i], float(sim[i])) for i in top_indices]


class ReferenceMatchWorker(QThread):
    """
    后台参考库匹配：以 labeled_persons（已知特征）为标准，向量化计算与所有人脸的相似度，
    按阈值匹配则标记对应人物，未达阈值则放入未知分组。与人脸归类不冲突：
    执行前会清除参考库/未知/未命名的旧结果，可于人脸归类之后重新匹配。
    """

    progress = Signal(int, int, str)  # current, total, stage_text
    finished_match = Signal(dict)     # {matched: int, unknown: int}
    error = Signal(str)

    def __init__(
        self,
        db: DatabaseManager,
        cosine_threshold: float = 0.60,
    ):
        super().__init__()
        self.db = db
        self.threshold = cosine_threshold
        self.logger = get_logger()
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        refs = self.db.get_labeled_persons_with_features()

        if not refs:
            self.logger.warning("参考库匹配: 无参考人物，请先导入参考库")
            self.finished_match.emit({"matched": 0, "unknown": 0})
            return

        # 与人脸归类一致：已有匹配结果时再次匹配，先清除旧结果从头开始
        cleared = self.db.clear_reference_match_results()
        if cleared > 0:
            self.logger.info(f"参考库匹配: 已清除 {cleared} 张人脸的旧匹配结果，从头开始")

        self.progress.emit(0, 1, "加载参考库特征...")
        ref_ids = [r["id"] for r in refs]
        ref_names = {r["id"]: r["person_id"] for r in refs}
        ref_matrix = np.vstack([
            DatabaseManager.feature_from_blob(r["feature"]).flatten().astype(np.float32)
            for r in refs
        ])
        norms = np.linalg.norm(ref_matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        ref_matrix = ref_matrix / norms  # (n_ref, 512)

        unknown_person_id = self.db.get_or_create_unknown_person()
        updates: list[tuple[int, int, float]] = []
        matched = 0
        unknown_count = 0
        processed = 0
        total_est = self.db.get_reference_matchable_face_count()

        for chunk_idx, faces in enumerate(self.db.iter_unassigned_faces_with_features(FACE_CHUNK_SIZE)):
            if self._cancelled:
                break
            self.progress.emit(0, max(total_est, 1), f"匹配人脸 {processed + 1}–{processed + len(faces)}...")

            face_ids = [f["id"] for f in faces]
            face_matrix = np.vstack([
                DatabaseManager.feature_from_blob(f["feature"]).flatten().astype(np.float32)
                for f in faces
            ])
            fnorms = np.linalg.norm(face_matrix, axis=1, keepdims=True)
            fnorms = np.maximum(fnorms, 1e-10)
            face_matrix = face_matrix / fnorms

            sim = ref_matrix @ face_matrix.T  # (n_ref, n_face)
            max_sim_per_face = np.max(sim, axis=0)
            best_ref_idx_per_face = np.argmax(sim, axis=0)

            for i, face_id in enumerate(face_ids):
                sim_val = float(max_sim_per_face[i])
                ref_idx = int(best_ref_idx_per_face[i])
                if sim_val >= self.threshold:
                    ref_label_id = ref_ids[ref_idx]
                    person_name = ref_names[ref_label_id]
                    person_id = self.db.get_or_create_person_by_name(person_name)
                    updates.append((person_id, face_id, sim_val))
                    matched += 1
                else:
                    updates.append((unknown_person_id, face_id, sim_val))
                    unknown_count += 1

            processed += len(faces)
            self.progress.emit(processed, max(total_est, 1), f"已匹配 {processed} 张...")

        if processed == 0:
            self.logger.info("参考库匹配: 无待匹配人脸")
            self.finished_match.emit({"matched": 0, "unknown": 0})
            return

        self.progress.emit(processed, max(total_est, 1), "写入数据库...")
        if updates:
            self.db.batch_update_face_persons_with_ref_similarity(updates)
            for person_id in set(p for p, _, _ in updates):
                count = self.db.get_person_face_count(person_id)
                self.db.update_person_face_count(person_id, count)

        self.logger.info(
            f"参考库匹配完成: 匹配 {matched} 张，未知 {unknown_count} 张，阈值={self.threshold:.2f}"
        )
        self.finished_match.emit({"matched": matched, "unknown": unknown_count})
