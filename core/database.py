"""SQLite 数据库管理模块"""

import sqlite3
import json
import numpy as np
from typing import Optional


class DatabaseManager:
    """人脸识别系统数据库管理器"""

    def __init__(self, db_path: str = "faceseeker.db"):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._auto_commit = True
        self._connect()
        self._init_db()

    def _connect(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")

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
                width INTEGER,
                height INTEGER,
                face_count INTEGER DEFAULT 0,
                added_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT DEFAULT '未命名',
                face_count INTEGER DEFAULT 0,
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
                FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
                FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE SET NULL
            );
        """)
        self.conn.commit()

    # ---- Images ----

    def image_exists(self, file_path: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM images WHERE file_path = ?", (file_path,)
        ).fetchone()
        return row is not None

    def add_image(self, file_path: str, filename: str, width: int, height: int, face_count: int = 0, relative_path: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO images (file_path, filename, relative_path, width, height, face_count) VALUES (?, ?, ?, ?, ?, ?)",
            (file_path, filename, relative_path, width, height, face_count),
        )
        self._maybe_commit()
        return cur.lastrowid

    def get_all_images(self) -> list:
        return self.conn.execute(
            "SELECT * FROM images ORDER BY added_time DESC"
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
    ) -> int:
        feature_blob = feature.tobytes() if feature is not None else None
        landmarks_json = json.dumps(landmarks)
        cur = self.conn.execute(
            "INSERT INTO faces (image_id, bbox_x, bbox_y, bbox_w, bbox_h, landmarks, score, feature, person_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, bbox[0], bbox[1], bbox[2], bbox[3], landmarks_json, score, feature_blob, person_id),
        )
        self._maybe_commit()
        return cur.lastrowid

    def get_faces_by_image(self, image_id: int) -> list:
        return self.conn.execute(
            "SELECT * FROM faces WHERE image_id = ?", (image_id,)
        ).fetchall()

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

    def get_faces_by_person(self, person_id: int) -> list:
        return self.conn.execute(
            "SELECT f.*, i.file_path, i.filename FROM faces f "
            "JOIN images i ON f.image_id = i.id "
            "WHERE f.person_id = ?",
            (person_id,),
        ).fetchall()

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

    def clear_all_persons(self):
        """清除所有人物归类（重新聚类前调用）"""
        self.conn.execute("UPDATE faces SET person_id = NULL")
        self.conn.execute("DELETE FROM persons")
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
            self.conn.close()
            self.conn = None
