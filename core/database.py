"""SQLite 数据库管理模块"""

import sqlite3
import json
import numpy as np
from typing import Optional

from core.logger import get_logger


class DatabaseManager:
    """人脸识别系统数据库管理器"""

    def __init__(self, db_path: str = "FaceAtlas.db"):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._auto_commit = True
        self.logger = get_logger()
        self.logger.info(f"开始连接数据库: {db_path}")
        self._connect()
        self._init_db()
        self._migrate()
        self.logger.info("数据库初始化完成")

    def _connect(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        # 性能优化 PRAGMA
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.execute("PRAGMA cache_size = -8000")    # 8 MB 页缓存
        self.conn.execute("PRAGMA temp_store = MEMORY")

    def _maybe_commit(self):
        """仅在非事务模式下自动提交"""
        if self._auto_commit:
            self.conn.commit()

    def _init_db(self):
        """建表"""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL UNIQUE,
                filename TEXT NOT NULL,
                relative_path TEXT,
                width INTEGER DEFAULT 0,
                height INTEGER DEFAULT 0,
                face_count INTEGER DEFAULT 0,
                analyzed INTEGER DEFAULT 0,
                added_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT DEFAULT '未命名',
                face_count INTEGER DEFAULT 0,
                feature BLOB,
                created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                bbox_x INTEGER,
                bbox_y INTEGER,
                bbox_w INTEGER,
                bbox_h INTEGER,
                landmarks TEXT,
                score REAL,
                feature BLOB,
                person_id INTEGER,
                blur_score REAL DEFAULT 0,
                FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
                FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE SET NULL
            );

            -- 索引加速聚类和查询
            CREATE INDEX IF NOT EXISTS idx_faces_image_id ON faces(image_id);
            CREATE INDEX IF NOT EXISTS idx_faces_person_id ON faces(person_id);
            CREATE INDEX IF NOT EXISTS idx_faces_has_feature ON faces(id) WHERE feature IS NOT NULL;
            CREATE INDEX IF NOT EXISTS idx_images_analyzed ON images(analyzed);
        """)
        self.conn.commit()

    def _migrate(self):
        """数据库迁移：为旧版数据库添加缺失的列"""
        img_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(images)").fetchall()}
        if "analyzed" not in img_columns:
            self.logger.info("数据库迁移：添加 analyzed 列")
            self.conn.execute("ALTER TABLE images ADD COLUMN analyzed INTEGER DEFAULT 0")
            self.conn.execute("UPDATE images SET analyzed = 1")
            self.conn.commit()

        face_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(faces)").fetchall()}
        if "blur_score" not in face_columns:
            self.logger.info("数据库迁移：添加 blur_score 列")
            self.conn.execute("ALTER TABLE faces ADD COLUMN blur_score REAL DEFAULT 0")
            self.conn.commit()

    # ---- Images ----

    def image_exists(self, file_path: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM images WHERE file_path = ?", (file_path,)
        ).fetchone()
        return row is not None

    def get_existing_paths(self) -> set[str]:
        """获取所有已注册的文件路径集合（用于批量去重，避免逐条查询）"""
        rows = self.conn.execute("SELECT file_path FROM images").fetchall()
        return {row[0] for row in rows}

    def add_image(self, file_path: str, filename: str, width: int = 0, height: int = 0,
                  face_count: int = 0, relative_path: str = "", analyzed: int = 0) -> int:
        cur = self.conn.execute(
            "INSERT INTO images (file_path, filename, relative_path, width, height, face_count, analyzed) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (file_path, filename, relative_path, width, height, face_count, analyzed),
        )
        self._maybe_commit()
        return cur.lastrowid

    def mark_image_analyzed(self, image_id: int, width: int, height: int, face_count: int):
        """标记图片为已分析，同时更新尺寸和人脸数"""
        self.conn.execute(
            "UPDATE images SET analyzed = 1, width = ?, height = ?, face_count = ? WHERE id = ?",
            (width, height, face_count, image_id),
        )
        self._maybe_commit()

    def get_image_count(self) -> int:
        """获取图片总数（轻量查询）"""
        row = self.conn.execute("SELECT COUNT(*) FROM images").fetchone()
        return row[0] if row else 0

    def get_analyzed_count(self) -> int:
        """获取已分析图片数量"""
        row = self.conn.execute("SELECT COUNT(*) FROM images WHERE analyzed = 1").fetchone()
        return row[0] if row else 0

    def get_unanalyzed_count(self) -> int:
        """获取未分析图片数量"""
        row = self.conn.execute("SELECT COUNT(*) FROM images WHERE analyzed = 0").fetchone()
        return row[0] if row else 0

    def get_unanalyzed_images(self) -> list:
        """获取所有未分析的图片"""
        return self.conn.execute(
            "SELECT * FROM images WHERE analyzed = 0 ORDER BY id"
        ).fetchall()

    def get_all_images(self) -> list:
        return self.conn.execute(
            "SELECT * FROM images ORDER BY added_time DESC"
        ).fetchall()

    def get_images_paginated(self, offset: int, limit: int) -> list:
        """分页获取图片列表（避免一次性加载全部）"""
        return self.conn.execute(
            "SELECT * FROM images ORDER BY added_time DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()

    def get_image(self, image_id: int):
        return self.conn.execute(
            "SELECT * FROM images WHERE id = ?", (image_id,)
        ).fetchone()

    def update_image_face_count(self, image_id: int, face_count: int):
        self.conn.execute(
            "UPDATE images SET face_count = ? WHERE id = ?", (face_count, image_id)
        )
        self._maybe_commit()

    def delete_image(self, image_id: int):
        self.conn.execute("DELETE FROM images WHERE id = ?", (image_id,))
        self._maybe_commit()

    # ---- Faces ----

    def add_face(
        self,
        image_id: int,
        bbox: tuple,
        landmarks: list,
        score: float,
        feature: Optional[np.ndarray] = None,
        person_id: Optional[int] = None,
        blur_score: float = 0.0,
    ) -> int:
        feature_blob = feature.tobytes() if feature is not None else None
        landmarks_json = json.dumps(landmarks)
        cur = self.conn.execute(
            "INSERT INTO faces (image_id, bbox_x, bbox_y, bbox_w, bbox_h, landmarks, score, feature, person_id, blur_score) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, bbox[0], bbox[1], bbox[2], bbox[3], landmarks_json, score, feature_blob, person_id, blur_score),
        )
        self._maybe_commit()
        return cur.lastrowid

    def add_faces_batch(self, faces: list[tuple]) -> list[int]:
        """批量插入人脸数据（在已有事务中调用效果最佳）"""
        ids = []
        for image_id, bbox, landmarks, score, feature, person_id in faces:
            feature_blob = feature.tobytes() if feature is not None else None
            landmarks_json = json.dumps(landmarks)
            cur = self.conn.execute(
                "INSERT INTO faces (image_id, bbox_x, bbox_y, bbox_w, bbox_h, landmarks, score, feature, person_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (image_id, bbox[0], bbox[1], bbox[2], bbox[3], landmarks_json, score, feature_blob, person_id),
            )
            ids.append(cur.lastrowid)
        self._maybe_commit()
        return ids

    def get_faces_by_image(self, image_id: int) -> list:
        return self.conn.execute(
            "SELECT * FROM faces WHERE image_id = ?", (image_id,)
        ).fetchall()

    def get_face_count(self) -> int:
        """获取人脸总数（轻量查询）"""
        row = self.conn.execute("SELECT COUNT(*) FROM faces WHERE feature IS NOT NULL").fetchone()
        return row[0] if row else 0

    def get_all_faces_with_features(self) -> list:
        """获取所有有特征向量的人脸，用于聚类"""
        return self.conn.execute(
            "SELECT * FROM faces WHERE feature IS NOT NULL"
        ).fetchall()

    def update_face_person(self, face_id: int, person_id: Optional[int]):
        self.conn.execute(
            "UPDATE faces SET person_id = ? WHERE id = ?", (person_id, face_id)
        )
        self._maybe_commit()

    def batch_update_face_persons(self, updates: list[tuple[int, int]]):
        """批量更新人脸的 person_id，每条为 (person_id, face_id)"""
        self.conn.executemany(
            "UPDATE faces SET person_id = ? WHERE id = ?", updates
        )
        self._maybe_commit()

    def delete_faces_by_image(self, image_id: int) -> list[int]:
        """删除指定图片的所有人脸数据，返回被删除的 face_id 列表"""
        rows = self.conn.execute(
            "SELECT id FROM faces WHERE image_id = ?", (image_id,)
        ).fetchall()
        face_ids = [row[0] for row in rows]
        if face_ids:
            self.conn.execute("DELETE FROM faces WHERE image_id = ?", (image_id,))
            self._maybe_commit()
        return face_ids

    def get_face(self, face_id: int):
        return self.conn.execute(
            "SELECT * FROM faces WHERE id = ?", (face_id,)
        ).fetchone()

    @staticmethod
    def feature_from_blob(blob: bytes) -> np.ndarray:
        """将 BLOB 转回 numpy 特征向量"""
        return np.frombuffer(blob, dtype=np.float32).copy()

    @staticmethod
    def landmarks_from_json(json_str: str) -> list:
        return json.loads(json_str)

    # ---- Persons ----

    def add_person(self, name: str = "未命名") -> int:
        cur = self.conn.execute(
            "INSERT INTO persons (name) VALUES (?)", (name,)
        )
        self._maybe_commit()
        return cur.lastrowid

    def get_all_persons(self) -> list:
        return self.conn.execute(
            "SELECT * FROM persons ORDER BY id"
        ).fetchall()

    def get_faces_by_person(self, person_id: int, limit: int = 0) -> list:
        """获取某人物的所有人脸（可选 LIMIT）"""
        if limit > 0:
            return self.conn.execute(
                "SELECT f.*, i.file_path, i.filename FROM faces f "
                "JOIN images i ON f.image_id = i.id "
                "WHERE f.person_id = ? LIMIT ?",
                (person_id, limit),
            ).fetchall()
        return self.conn.execute(
            "SELECT f.*, i.file_path, i.filename FROM faces f "
            "JOIN images i ON f.image_id = i.id "
            "WHERE f.person_id = ?",
            (person_id,),
        ).fetchall()

    def get_person_face_count(self, person_id: int) -> int:
        """获取某人物的人脸总数（轻量查询）"""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM faces WHERE person_id = ?", (person_id,)
        ).fetchone()
        return row[0] if row else 0

    def update_person_name(self, person_id: int, name: str):
        self.conn.execute(
            "UPDATE persons SET name = ? WHERE id = ?", (name, person_id)
        )
        self._maybe_commit()

    def update_person_face_count(self, person_id: int, count: int):
        self.conn.execute(
            "UPDATE persons SET face_count = ? WHERE id = ?", (count, person_id)
        )
        self._maybe_commit()

    def update_person_feature(self, person_id: int, feature: Optional[np.ndarray]):
        """更新人物的代表性特征向量"""
        feature_blob = feature.tobytes() if feature is not None else None
        self.conn.execute(
            "UPDATE persons SET feature = ? WHERE id = ?", (feature_blob, person_id)
        )
        self._maybe_commit()

    def get_person_feature(self, person_id: int) -> Optional[np.ndarray]:
        """获取人物的代表性特征向量"""
        row = self.conn.execute(
            "SELECT feature FROM persons WHERE id = ?", (person_id,)
        ).fetchone()
        if row and row["feature"]:
            return self.feature_from_blob(row["feature"])
        return None

    def get_persons_with_features(self) -> list:
        """获取所有有代表性特征的人物"""
        return self.conn.execute(
            "SELECT * FROM persons WHERE feature IS NOT NULL"
        ).fetchall()

    def get_unassigned_faces_with_features(self) -> list:
        """获取所有未分配人物的人脸（有特征向量）"""
        return self.conn.execute(
            "SELECT * FROM faces WHERE feature IS NOT NULL AND person_id IS NULL"
        ).fetchall()

    def clear_all_persons(self, keep_named: bool = False):
        """清除人物归类"""
        if keep_named:
            self.conn.execute(
                "UPDATE faces SET person_id = NULL WHERE person_id IN "
                "(SELECT id FROM persons WHERE name = '未命名')"
            )
            self.conn.execute("DELETE FROM persons WHERE name = '未命名'")
        else:
            self.conn.execute("UPDATE faces SET person_id = NULL")
            self.conn.execute("DELETE FROM persons")
            # 重置自增计数器，使新的 person_id 从 1 开始
            self.conn.execute("DELETE FROM sqlite_sequence WHERE name = 'persons'")
        self._maybe_commit()

    def delete_all_images(self):
        """高效清空所有图片和关联数据"""
        self.conn.execute("DELETE FROM faces")
        self.conn.execute("DELETE FROM images")
        self._maybe_commit()

    # ---- Transaction ----

    def begin(self):
        """开启事务，抑制各方法的自动 commit"""
        self._auto_commit = False

    def commit(self):
        """提交事务，恢复自动 commit"""
        self.conn.commit()
        self._auto_commit = True

    def rollback(self):
        """回滚事务，恢复自动 commit"""
        self.conn.rollback()
        self._auto_commit = True

    # ---- Cleanup ----

    def close(self):
        if self.conn:
            self.logger.info("关闭数据库连接")
            self.conn.close()
            self.conn = None
