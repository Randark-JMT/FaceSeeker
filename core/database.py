"""PostgreSQL 数据库管理模块"""

import json
import threading
from typing import Any, Optional

import numpy as np
import psycopg2
from psycopg2 import sql

try:
    from pgvector.psycopg2 import register_vector
except ImportError:
    register_vector = None
from psycopg2.extras import RealDictCursor

from core.logger import get_logger


def _to_native(obj: Any) -> Any:
    """递归将 numpy 标量/数组转为 Python 原生类型，避免 psycopg2/JSON 兼容问题。"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    return obj


class DatabaseManager:
    """人脸识别系统数据库管理器（PostgreSQL）"""

    EXPECTED_TABLES = {"faceatlas_meta", "images", "persons", "faces"}

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
    ):
        self.host = host
        self.port = int(port)
        self.user = user
        self.password = password
        self.database = database
        self.conn: Optional[psycopg2.extensions.connection] = None
        self._lock = threading.RLock()
        self._auto_commit = True
        self._pgvector_available = False
        self.logger = get_logger()

        self.logger.info(f"开始连接数据库: {host}:{port}/{database}")
        self._connect()
        self._init_db()
        self._migrate()
        self.logger.info("数据库初始化完成")

    # ---- 连接与校验 ----

    @staticmethod
    def list_databases(host: str, port: int, user: str, password: str) -> list[str]:
        conn = psycopg2.connect(
            host=host,
            port=int(port),
            user=user,
            password=password,
            dbname="postgres",
        )
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT datname
                    FROM pg_database
                    WHERE datistemplate = FALSE
                    ORDER BY datname
                    """
                )
                return [row[0] for row in cur.fetchall()]
        finally:
            conn.close()

    @staticmethod
    def ensure_database_exists(host: str, port: int, user: str, password: str, database: str) -> bool:
        conn = psycopg2.connect(
            host=host,
            port=int(port),
            user=user,
            password=password,
            dbname="postgres",
        )
        try:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database,))
                exists = cur.fetchone() is not None
                if not exists:
                    cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
                return not exists
        finally:
            conn.close()

    @classmethod
    def validate_database_schema(
        cls, host: str, port: int, user: str, password: str, database: str
    ) -> tuple[bool, str]:
        conn = psycopg2.connect(
            host=host,
            port=int(port),
            user=user,
            password=password,
            dbname=database,
        )
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                    """
                )
                tables = {row[0] for row in cur.fetchall()}
        finally:
            conn.close()

        if not tables:
            return True, "空数据库，将自动初始化 FaceAtlas 表结构。"

        unexpected = sorted(tables - cls.EXPECTED_TABLES)
        if unexpected:
            return (
                False,
                "数据库中存在非 FaceAtlas 业务表，拒绝连接："
                + ", ".join(unexpected),
            )

        return True, "数据库结构合法。"

    # ---- 内部 SQL 工具 ----

    def _connect(self):
        self.conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            dbname=self.database,
        )
        self.conn.autocommit = False
        if register_vector is not None:
            register_vector(self.conn)

    def _execute(self, query: str, params=None):
        with self._lock:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)

    def _execute_fetchone(self, query: str, params=None):
        with self._lock:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                return cur.fetchone()

    def _execute_fetchall(self, query: str, params=None) -> list:
        with self._lock:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                return cur.fetchall()

    def _execute_returning_id(self, query: str, params=None) -> int:
        with self._lock:
            with self.conn.cursor() as cur:
                cur.execute(query, params)
                row = cur.fetchone()
        return int(row[0])

    def _executemany(self, query: str, params_seq):
        with self._lock:
            with self.conn.cursor() as cur:
                cur.executemany(query, params_seq)

    def _maybe_commit(self):
        if self._auto_commit:
            with self._lock:
                self.conn.commit()

    def _init_db(self):
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS faceatlas_meta (
                id SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
                schema_version INTEGER NOT NULL,
                created_time TIMESTAMP NOT NULL DEFAULT NOW()
            )
            """
        )
        self._execute(
            """
            INSERT INTO faceatlas_meta (id, schema_version)
            VALUES (1, 1)
            ON CONFLICT (id) DO NOTHING
            """
        )
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                id BIGSERIAL PRIMARY KEY,
                file_path TEXT NOT NULL UNIQUE,
                filename TEXT NOT NULL,
                relative_path TEXT,
                width INTEGER DEFAULT 0,
                height INTEGER DEFAULT 0,
                face_count INTEGER DEFAULT 0,
                analyzed INTEGER DEFAULT 0,
                added_time TIMESTAMP NOT NULL DEFAULT NOW()
            )
            """
        )
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS persons (
                id BIGSERIAL PRIMARY KEY,
                name TEXT DEFAULT '未命名',
                face_count INTEGER DEFAULT 0,
                feature BYTEA,
                created_time TIMESTAMP NOT NULL DEFAULT NOW()
            )
            """
        )
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS faces (
                id BIGSERIAL PRIMARY KEY,
                image_id BIGINT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
                bbox_x INTEGER,
                bbox_y INTEGER,
                bbox_w INTEGER,
                bbox_h INTEGER,
                landmarks TEXT,
                score DOUBLE PRECISION,
                feature BYTEA,
                person_id BIGINT REFERENCES persons(id) ON DELETE SET NULL,
                blur_score DOUBLE PRECISION DEFAULT 0
            )
            """
        )
        self._execute("CREATE INDEX IF NOT EXISTS idx_faces_image_id ON faces(image_id)")
        self._execute("CREATE INDEX IF NOT EXISTS idx_faces_person_id ON faces(person_id)")
        self._execute("CREATE INDEX IF NOT EXISTS idx_images_analyzed ON images(analyzed)")
        self._execute("CREATE INDEX IF NOT EXISTS idx_faces_has_feature ON faces(id) WHERE feature IS NOT NULL")
        with self._lock:
            self.conn.commit()

    def _migrate(self):
        rows = self._execute_fetchall(
            """
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name IN ('images', 'faces', 'persons')
            """
        )
        columns = {(r["table_name"], r["column_name"]) for r in rows}
        if ("images", "analyzed") not in columns:
            self.logger.info("数据库迁移：添加 analyzed 列")
            self._execute("ALTER TABLE images ADD COLUMN analyzed INTEGER DEFAULT 0")
            self._execute("UPDATE images SET analyzed = 1")
        if ("faces", "blur_score") not in columns:
            self.logger.info("数据库迁移：添加 blur_score 列")
            self._execute("ALTER TABLE faces ADD COLUMN blur_score DOUBLE PRECISION DEFAULT 0")

        # pgvector 迁移
        self._migrate_pgvector(columns)

        with self._lock:
            self.conn.commit()

    def _migrate_pgvector(self, columns: set[tuple[str, str]]):
        """添加 pgvector 扩展、embedding 列、索引及回填。"""
        try:
            self._execute("CREATE EXTENSION IF NOT EXISTS vector")
        except Exception as e:
            self.logger.warning("pgvector 扩展不可用，跳过向量索引迁移: %s", e)
            return
        self._pgvector_available = True

        if ("faces", "embedding") not in columns:
            self.logger.info("数据库迁移：添加 faces.embedding 列")
            self._execute("ALTER TABLE faces ADD COLUMN embedding vector(512)")
        if ("persons", "embedding") not in columns:
            self.logger.info("数据库迁移：添加 persons.embedding 列")
            self._execute("ALTER TABLE persons ADD COLUMN embedding vector(512)")

        # 回填 faces：从 feature BYTEA 填充 embedding
        rows = self._execute_fetchall(
            "SELECT id, feature FROM faces WHERE feature IS NOT NULL AND embedding IS NULL"
        )
        if rows:
            self.logger.info("回填 faces.embedding: %d 条", len(rows))
            for row in rows:
                arr = self.feature_from_blob(row["feature"]).flatten().astype(np.float32)
                self._execute("UPDATE faces SET embedding = %s WHERE id = %s", (arr, row["id"]))

        # 回填 persons
        rows = self._execute_fetchall(
            "SELECT id, feature FROM persons WHERE feature IS NOT NULL AND embedding IS NULL"
        )
        if rows:
            self.logger.info("回填 persons.embedding: %d 条", len(rows))
            for row in rows:
                arr = self.feature_from_blob(row["feature"]).flatten().astype(np.float32)
                self._execute("UPDATE persons SET embedding = %s WHERE id = %s", (arr, row["id"]))

        # 创建 HNSW 索引（若不存在）
        idx_rows = self._execute_fetchall(
            """
            SELECT indexname FROM pg_indexes
            WHERE schemaname = 'public' AND indexname = 'idx_faces_embedding_hnsw'
            """
        )
        if not idx_rows:
            self.logger.info("创建 faces.embedding HNSW 索引")
            self._execute(
                """
                CREATE INDEX idx_faces_embedding_hnsw ON faces
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
                WHERE embedding IS NOT NULL
                """
            )
        idx_rows = self._execute_fetchall(
            """
            SELECT indexname FROM pg_indexes
            WHERE schemaname = 'public' AND indexname = 'idx_persons_embedding_hnsw'
            """
        )
        if not idx_rows:
            self.logger.info("创建 persons.embedding HNSW 索引")
            self._execute(
                """
                CREATE INDEX idx_persons_embedding_hnsw ON persons
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
                WHERE embedding IS NOT NULL
                """
            )

    # ---- Images ----

    def image_exists(self, file_path: str) -> bool:
        row = self._execute_fetchone("SELECT 1 AS ok FROM images WHERE file_path = %s", (file_path,))
        return row is not None

    def get_existing_paths(self) -> set[str]:
        rows = self._execute_fetchall("SELECT file_path FROM images")
        return {row["file_path"] for row in rows}

    def add_image(
        self,
        file_path: str,
        filename: str,
        width: int = 0,
        height: int = 0,
        face_count: int = 0,
        relative_path: str = "",
        analyzed: int = 0,
    ) -> int:
        image_id = self._execute_returning_id(
            """
            INSERT INTO images (file_path, filename, relative_path, width, height, face_count, analyzed)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (file_path, filename, relative_path, width, height, face_count, analyzed),
        )
        self._maybe_commit()
        return image_id

    def mark_image_analyzed(self, image_id: int, width: int, height: int, face_count: int):
        self._execute(
            "UPDATE images SET analyzed = 1, width = %s, height = %s, face_count = %s WHERE id = %s",
            (width, height, face_count, image_id),
        )
        self._maybe_commit()

    def get_image_count(self) -> int:
        row = self._execute_fetchone("SELECT COUNT(*) AS cnt FROM images")
        return int(row["cnt"]) if row else 0

    def get_analyzed_count(self) -> int:
        row = self._execute_fetchone("SELECT COUNT(*) AS cnt FROM images WHERE analyzed = 1")
        return int(row["cnt"]) if row else 0

    def get_unanalyzed_count(self) -> int:
        row = self._execute_fetchone("SELECT COUNT(*) AS cnt FROM images WHERE analyzed = 0")
        return int(row["cnt"]) if row else 0

    def get_unanalyzed_images(self) -> list:
        return self._execute_fetchall("SELECT * FROM images WHERE analyzed = 0 ORDER BY id")

    def get_all_images(self) -> list:
        return self._execute_fetchall("SELECT * FROM images ORDER BY added_time DESC")

    def get_images_paginated(self, offset: int, limit: int) -> list:
        return self._execute_fetchall(
            "SELECT * FROM images ORDER BY added_time DESC LIMIT %s OFFSET %s",
            (limit, offset),
        )

    def get_image(self, image_id: int):
        return self._execute_fetchone("SELECT * FROM images WHERE id = %s", (image_id,))

    def update_image_face_count(self, image_id: int, face_count: int):
        self._execute("UPDATE images SET face_count = %s WHERE id = %s", (face_count, image_id))
        self._maybe_commit()

    def delete_image(self, image_id: int):
        self._execute("DELETE FROM images WHERE id = %s", (image_id,))
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
        embedding_arr = (
            feature.flatten().astype(np.float32)
            if self._pgvector_available and feature is not None and feature.size > 0
            else None
        )
        landmarks_json = json.dumps(_to_native(landmarks))
        if self._pgvector_available:
            face_id = self._execute_returning_id(
                """
                INSERT INTO faces (image_id, bbox_x, bbox_y, bbox_w, bbox_h, landmarks, score, feature, embedding, person_id, blur_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    int(image_id),
                    int(bbox[0]),
                    int(bbox[1]),
                    int(bbox[2]),
                    int(bbox[3]),
                    landmarks_json,
                    float(score),
                    feature_blob,
                    embedding_arr,
                    person_id,
                    float(blur_score),
                ),
            )
        else:
            face_id = self._execute_returning_id(
                """
                INSERT INTO faces (image_id, bbox_x, bbox_y, bbox_w, bbox_h, landmarks, score, feature, person_id, blur_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    int(image_id),
                    int(bbox[0]),
                    int(bbox[1]),
                    int(bbox[2]),
                    int(bbox[3]),
                    landmarks_json,
                    float(score),
                    feature_blob,
                    person_id,
                    float(blur_score),
                ),
            )
        self._maybe_commit()
        return face_id

    def add_faces_batch(self, faces: list[tuple]) -> list[int]:
        ids = []
        for image_id, bbox, landmarks, score, feature, person_id in faces:
            feature_blob = feature.tobytes() if feature is not None else None
            embedding_arr = (
                feature.flatten().astype(np.float32)
                if self._pgvector_available and feature is not None and feature.size > 0
                else None
            )
            landmarks_json = json.dumps(_to_native(landmarks))
            if self._pgvector_available:
                face_id = self._execute_returning_id(
                    """
                    INSERT INTO faces (image_id, bbox_x, bbox_y, bbox_w, bbox_h, landmarks, score, feature, embedding, person_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        int(image_id),
                        int(bbox[0]),
                        int(bbox[1]),
                        int(bbox[2]),
                        int(bbox[3]),
                        landmarks_json,
                        float(score),
                        feature_blob,
                        embedding_arr,
                        person_id,
                    ),
                )
            else:
                face_id = self._execute_returning_id(
                    """
                    INSERT INTO faces (image_id, bbox_x, bbox_y, bbox_w, bbox_h, landmarks, score, feature, person_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        int(image_id),
                        int(bbox[0]),
                        int(bbox[1]),
                        int(bbox[2]),
                        int(bbox[3]),
                        landmarks_json,
                        float(score),
                        feature_blob,
                        person_id,
                    ),
                )
            ids.append(face_id)
        self._maybe_commit()
        return ids

    def get_faces_by_image(self, image_id: int) -> list:
        return self._execute_fetchall("SELECT * FROM faces WHERE image_id = %s", (image_id,))

    def get_face_count(self) -> int:
        row = self._execute_fetchone("SELECT COUNT(*) AS cnt FROM faces WHERE feature IS NOT NULL")
        return int(row["cnt"]) if row else 0

    def get_all_faces_with_features(self) -> list:
        return self._execute_fetchall("SELECT * FROM faces WHERE feature IS NOT NULL")

    def update_face_person(self, face_id: int, person_id: Optional[int]):
        self._execute("UPDATE faces SET person_id = %s WHERE id = %s", (person_id, face_id))
        self._maybe_commit()

    def batch_update_face_persons(self, updates: list[tuple[int, int]]):
        self._executemany("UPDATE faces SET person_id = %s WHERE id = %s", updates)
        self._maybe_commit()

    def delete_faces_by_image(self, image_id: int) -> list[int]:
        rows = self._execute_fetchall("SELECT id FROM faces WHERE image_id = %s", (image_id,))
        face_ids = [int(row["id"]) for row in rows]
        if face_ids:
            self._execute("DELETE FROM faces WHERE image_id = %s", (image_id,))
            self._maybe_commit()
        return face_ids

    def get_face(self, face_id: int):
        return self._execute_fetchone("SELECT * FROM faces WHERE id = %s", (face_id,))

    @staticmethod
    def feature_from_blob(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32).copy()

    @staticmethod
    def landmarks_from_json(json_str: str) -> list:
        return json.loads(json_str)

    # ---- Persons ----

    def add_person(self, name: str = "未命名") -> int:
        person_id = self._execute_returning_id(
            "INSERT INTO persons (name) VALUES (%s) RETURNING id", (name,)
        )
        self._maybe_commit()
        return person_id

    def get_all_persons(self) -> list:
        return self._execute_fetchall("SELECT * FROM persons ORDER BY id")

    def get_faces_by_person(self, person_id: int, limit: int = 0) -> list:
        if limit > 0:
            return self._execute_fetchall(
                """
                SELECT f.*, i.file_path, i.filename
                FROM faces f
                JOIN images i ON f.image_id = i.id
                WHERE f.person_id = %s
                ORDER BY f.id
                LIMIT %s
                """,
                (person_id, limit),
            )
        return self._execute_fetchall(
            """
            SELECT f.*, i.file_path, i.filename
            FROM faces f
            JOIN images i ON f.image_id = i.id
            WHERE f.person_id = %s
            ORDER BY f.id
            """,
            (person_id,),
        )

    def get_person_face_count(self, person_id: int) -> int:
        row = self._execute_fetchone("SELECT COUNT(*) AS cnt FROM faces WHERE person_id = %s", (person_id,))
        return int(row["cnt"]) if row else 0

    def update_person_name(self, person_id: int, name: str):
        self._execute("UPDATE persons SET name = %s WHERE id = %s", (name, person_id))
        self._maybe_commit()

    def update_person_face_count(self, person_id: int, count: int):
        self._execute("UPDATE persons SET face_count = %s WHERE id = %s", (count, person_id))
        self._maybe_commit()

    def update_person_feature(self, person_id: int, feature: Optional[np.ndarray]):
        feature_blob = feature.tobytes() if feature is not None else None
        if self._pgvector_available and feature is not None and feature.size > 0:
            embedding_arr = feature.flatten().astype(np.float32)
            self._execute(
                "UPDATE persons SET feature = %s, embedding = %s WHERE id = %s",
                (feature_blob, embedding_arr, person_id),
            )
        else:
            self._execute("UPDATE persons SET feature = %s WHERE id = %s", (feature_blob, person_id))
        self._maybe_commit()

    def get_person_feature(self, person_id: int) -> Optional[np.ndarray]:
        row = self._execute_fetchone("SELECT feature FROM persons WHERE id = %s", (person_id,))
        if row and row["feature"]:
            return self.feature_from_blob(row["feature"])
        return None

    def get_persons_with_features(self) -> list:
        return self._execute_fetchall("SELECT * FROM persons WHERE feature IS NOT NULL")

    def get_unassigned_faces_with_features(self) -> list:
        return self._execute_fetchall("SELECT * FROM faces WHERE feature IS NOT NULL AND person_id IS NULL")

    def find_similar_person_for_face(
        self,
        query_feature: np.ndarray,
        cosine_threshold: float = 0.60,
    ) -> Optional[tuple[int, float]]:
        """
        使用 pgvector K-NN 查找与 query_feature 最相似的已有人物。
        余弦距离 = 1 - 余弦相似度，故 threshold 0.6 对应 distance 0.4。
        需 pgvector 可用且 persons 表有 embedding 列。
        """
        if not self._pgvector_available:
            return None
        arr = query_feature.flatten().astype(np.float32)
        norm = np.linalg.norm(arr)
        if norm < 1e-10:
            return None
        arr = arr / norm
        cos_distance_max = 1.0 - cosine_threshold
        row = self._execute_fetchone(
            """
            SELECT id, 1 - (embedding <=> %s) AS similarity
            FROM persons
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s
            LIMIT 1
            """,
            (arr, arr),
        )
        if row and row["similarity"] >= cosine_threshold:
            return (int(row["id"]), float(row["similarity"]))
        return None

    def find_similar_faces(
        self,
        query_feature: np.ndarray,
        limit: int = 10,
        cosine_threshold: float = 0.0,
    ) -> list[dict]:
        """
        使用 pgvector K-NN 查找与 query_feature 相似的人脸，按相似度降序。
        可用于 UI「找相似人脸」功能。
        """
        if not self._pgvector_available:
            return []
        arr = query_feature.flatten().astype(np.float32)
        norm = np.linalg.norm(arr)
        if norm < 1e-10:
            return []
        arr = arr / norm
        return self._execute_fetchall(
            """
            SELECT f.id, f.image_id, f.person_id, 1 - (f.embedding <=> %s) AS similarity
            FROM faces f
            WHERE f.embedding IS NOT NULL
              AND (1 - (f.embedding <=> %s)) >= %s
            ORDER BY f.embedding <=> %s
            LIMIT %s
            """,
            (arr, arr, cosine_threshold, arr, limit),
        )

    def clear_all_persons(self, keep_named: bool = False):
        if keep_named:
            self._execute(
                """
                UPDATE faces SET person_id = NULL
                WHERE person_id IN (SELECT id FROM persons WHERE name = '未命名')
                """
            )
            self._execute("DELETE FROM persons WHERE name = '未命名'")
        else:
            self._execute("UPDATE faces SET person_id = NULL")
            self._execute("DELETE FROM persons")
            self._execute("ALTER SEQUENCE persons_id_seq RESTART WITH 1")
        self._maybe_commit()

    def delete_all_images(self):
        self._execute("DELETE FROM faces")
        self._execute("DELETE FROM images")
        self._maybe_commit()

    # ---- Transaction ----

    def begin(self):
        self._auto_commit = False

    def commit(self):
        with self._lock:
            self.conn.commit()
        self._auto_commit = True

    def rollback(self):
        with self._lock:
            self.conn.rollback()
        self._auto_commit = True

    # ---- Cleanup ----

    def close(self):
        if self.conn:
            self.logger.info("关闭数据库连接")
            with self._lock:
                self.conn.close()
            self.conn = None
