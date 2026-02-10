"""人脸聚类模块 - 基于向量化余弦相似度 + Union-Find 将相似人脸归为同一人"""

from collections import defaultdict

import numpy as np

from core.database import DatabaseManager


class UnionFind:
    """并查集（路径压缩 + 按秩合并）"""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # 路径压半
            x = self.parent[x]
        return x

    def union(self, x: int, y: int):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


class FaceCluster:
    """人脸聚类器（向量化加速版）"""

    # 分块大小：控制内存占用，每块最多占 BLOCK * n * 4 字节
    BLOCK_SIZE = 512

    def __init__(self, db: DatabaseManager, recognizer=None):
        self.db = db
        # recognizer 保留兼容，但不再用于逐对 match
        self.recognizer = recognizer

    def cluster(self, cosine_threshold: float = 0.363,
                progress_cb=None) -> dict[int, list[int]]:
        """
        对数据库中所有有特征的人脸进行聚类。
        使用 numpy 向量化矩阵乘法计算余弦相似度，性能远优于逐对调用。

        Args:
            cosine_threshold: 余弦相似度阈值（SFace 推荐 0.363）
            progress_cb: 进度回调 (current, total, stage_text)

        Returns:
            {person_id: [face_id, ...]} 聚类结果
        """

        def _report(current, total, text):
            if progress_cb:
                progress_cb(current, total, text)

        # 清除旧的归类
        _report(0, 1, "清除旧归类数据...")
        self.db.clear_all_persons()

        rows = self.db.get_all_faces_with_features()
        if not rows:
            return {}

        # 加载 face_id 和特征
        _report(0, 1, "加载人脸特征...")
        face_ids: list[int] = []
        feat_list: list[np.ndarray] = []
        for row in rows:
            face_ids.append(row["id"])
            feat = DatabaseManager.feature_from_blob(row["feature"])
            feat_list.append(feat.flatten())

        n = len(face_ids)

        # 构建特征矩阵 (n, dim) 并 L2 归一化
        feat_matrix = np.vstack(feat_list).astype(np.float32)  # (n, dim)
        norms = np.linalg.norm(feat_matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # 避免除零
        feat_matrix /= norms

        _report(0, n, f"加载 {n} 张人脸，开始向量化比对...")

        # Union-Find 聚类（基于索引 0..n-1）
        uf = UnionFind(n)

        # 分块计算相似度矩阵，避免一次性分配 n*n 矩阵导致内存不足
        block = self.BLOCK_SIZE
        total_blocks = (n + block - 1) // block
        processed_blocks = 0

        for i_start in range(0, n, block):
            i_end = min(i_start + block, n)
            # 只计算上三角部分：j >= i_start
            # 对于当前块的行 [i_start, i_end)，与所有列 [i_start, n) 比较
            chunk_i = feat_matrix[i_start:i_end]          # (block_i, dim)
            chunk_j = feat_matrix[i_start:]               # (n - i_start, dim)
            sim_block = chunk_i @ chunk_j.T               # (block_i, n - i_start)

            # 提取超过阈值的配对
            rows_idx, cols_idx = np.where(sim_block >= cosine_threshold)
            for r, c in zip(rows_idx, cols_idx):
                abs_i = i_start + r
                abs_j = i_start + c
                if abs_i < abs_j:  # 只取上三角
                    uf.union(abs_i, abs_j)

            processed_blocks += 1
            _report(processed_blocks, total_blocks,
                    f"比对进度: 第 {processed_blocks}/{total_blocks} 块 "
                    f"(行 {i_start}-{i_end-1}/{n-1})")

        # 收集分组（索引 → face_id）
        groups: dict[int, list[int]] = defaultdict(list)
        for idx in range(n):
            root = uf.find(idx)
            groups[root].append(face_ids[idx])

        _report(total_blocks, total_blocks,
                f"比对完成，正在写入 {len(groups)} 个人物分组...")

        # 写入数据库（单事务批量提交）
        result: dict[int, list[int]] = {}
        self.db.begin()
        try:
            for group_face_ids in groups.values():
                person_id = self.db.add_person()
                self.db.update_person_face_count(person_id, len(group_face_ids))
                for fid in group_face_ids:
                    self.db.update_face_person(fid, person_id)
                result[person_id] = group_face_ids
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

        return result
